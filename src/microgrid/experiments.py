"""Instance generation and the three experiments used in the paper."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from .core import Instance, Solution, check_solution, quantize, solve_algorithm_1, solve_algorithm_1_with_sales, solve_algorithm_2, solve_gurobi


ALGORITHM_1 = "Algorithm 1 + Theorem 5"
ALGORITHM_2 = "Algorithm 2 O(nC)"
NO_STORAGE = "No storage"
SELF_CONSUMPTION = "Myopic self-consumption"


def _instance(window, capacity_ratio, pv_ratio, unit_kwh, sell_price, name_suffix="", buy_price=None):
    demand_kwh = window.demand_kwh.to_numpy(float)
    annual_load = float(window.attrs["annual_load_kwh"])
    pv_kwp = pv_ratio * annual_load / float(window.attrs["annual_pv_yield"])
    pv_kwh = window.pv_kwh_per_kwp.to_numpy(float) * pv_kwp
    demand, pv = quantize(demand_kwh, unit_kwh), quantize(pv_kwh, unit_kwh)
    capacity = int(round(capacity_ratio * annual_load / 365 / unit_kwh))
    residual, excess = np.maximum(demand - pv, 0), np.maximum(pv - demand, 0)
    household, season = window.attrs["household_id"], window.attrs["season"]
    return Instance(
        demand, pv, window.buy_price.to_numpy(float) if buy_price is None else np.asarray(buy_price, float), np.asarray(sell_price, float), capacity, 0,
        residual, capacity + excess, unit_kwh,
        f"{household}_{season}_{name_suffix}",
        {"household_id": household, "season": season, "annual_load_kwh": annual_load,
         "capacity_ratio": capacity_ratio, "capacity_kwh": capacity * unit_kwh,
         "pv_ratio": pv_ratio, "pv_kwp": pv_kwp,
         "demand_rounding_kwh": demand.sum() * unit_kwh - demand_kwh.sum(),
         "pv_rounding_kwh": pv.sum() * unit_kwh - pv_kwh.sum()},
    )


def _assets(windows, config):
    households = sorted({window.attrs["household_id"] for window in windows})
    rng, case = np.random.default_rng(config["seed"] + 19), config["case_study"]
    adopters = set(rng.choice(households, int(round(case["pv_adoption_rate"] * len(households))), replace=False))
    rows = []
    for household in households:
        pv_ratio = float(np.clip(rng.lognormal(np.log(case["median_pv_ratio"]), case["asset_log_sigma"]), 0.25, 2.0)) if household in adopters else 0.0
        capacity_ratio = float(np.clip(rng.lognormal(np.log(case["median_capacity_ratio"]), case["asset_log_sigma"]), 0.25, 2.0))
        rows.append({"household_id": household, "pv_ratio": pv_ratio, "capacity_ratio": capacity_ratio})
    return pd.DataFrame(rows).set_index("household_id")


def _microgrid_prices(windows, assets, floor, ceiling, elasticity):
    prices = {}
    for season in sorted({window.attrs["season"] for window in windows}):
        group = [window for window in windows if window.attrs["season"] == season]
        demand = sum(window.demand_kwh.to_numpy(float) for window in group)
        pv = sum(window.pv_kwh_per_kwp.to_numpy(float) * assets.loc[window.attrs["household_id"], "pv_ratio"] * window.attrs["annual_load_kwh"] / window.attrs["annual_pv_yield"] for window in group)
        scarcity = demand / np.maximum(demand + pv, 1e-12)
        buy = group[0].buy_price.to_numpy(float) * np.exp(elasticity * (scarcity - np.median(scarcity)))
        ratio = floor + (ceiling - floor) * scarcity
        prices[season] = buy, buy * ratio
    return prices


def _no_storage(instance):
    purchase, sale = np.maximum(instance.demand - instance.pv, 0).astype(float), np.maximum(instance.pv - instance.demand, 0).astype(float)
    objective = instance.unit_kwh * (np.dot(instance.buy_price, purchase) - np.dot(instance.sell_price, sale))
    return Solution(float(objective), purchase, sale, np.zeros(instance.n), algorithm=NO_STORAGE)


def _self_consumption(instance):
    purchase, sale, soc = np.zeros((3, instance.n))
    state = instance.initial_soc
    for period, net in enumerate(instance.pv - instance.demand):
        if net >= 0:
            charge = min(instance.capacity - state, int(net))
            state += charge
            sale[period] = net - charge
        else:
            discharge = min(state, int(-net))
            state -= discharge
            purchase[period] = -net - discharge
        soc[period] = state
    objective = instance.unit_kwh * (np.dot(instance.buy_price, purchase) - np.dot(instance.sell_price, sale))
    solution = Solution(float(objective), purchase, sale, soc, algorithm=SELF_CONSUMPTION)
    check_solution(instance, solution)
    return solution


def _metrics(instance, solution, policy, scale, no_storage, self_consumption=None):
    gross_import = instance.unit_kwh * float(np.dot(instance.buy_price, np.maximum(instance.demand - instance.pv, 0)))
    purchase_kwh, sale_kwh = instance.unit_kwh * solution.purchase.sum(), instance.unit_kwh * solution.sale.sum()
    demand_kwh, pv_kwh = instance.unit_kwh * instance.demand.sum(), instance.unit_kwh * instance.pv.sum()
    return {**instance.metadata, "instance": instance.name, "policy": policy, "capacity_scale": scale,
            "net_bill_gbp": solution.objective, "no_storage_bill_gbp": no_storage.objective,
            "benefit_vs_no_storage_gbp": no_storage.objective - solution.objective,
            "benefit_vs_no_storage_percent": 100 * (no_storage.objective - solution.objective) / gross_import if gross_import else np.nan,
            "benefit_vs_self_consumption_gbp": self_consumption.objective - solution.objective if self_consumption else np.nan,
            "benefit_vs_self_consumption_percent": 100 * (self_consumption.objective - solution.objective) / gross_import if self_consumption and gross_import else np.nan,
            "gross_import_cost_gbp": gross_import, "purchase_kwh": purchase_kwh, "sale_kwh": sale_kwh,
            "demand_kwh": demand_kwh, "pv_kwh": pv_kwh,
            "self_sufficiency": 1 - purchase_kwh / demand_kwh if demand_kwh else np.nan,
            "battery_throughput_kwh": instance.unit_kwh * np.abs(np.diff(np.r_[instance.initial_soc, solution.soc])).sum(),
            "final_soc_kwh": instance.unit_kwh * solution.soc[-1]}


def _schedule(instance, solution, policy, scale, index, price_scenario):
    return pd.DataFrame({"timestamp": index, "household_id": instance.metadata["household_id"], "season": instance.metadata["season"],
                         "price_scenario": price_scenario, "policy": policy, "capacity_scale": scale, "demand_kwh": instance.demand * instance.unit_kwh,
                         "pv_kwh": instance.pv * instance.unit_kwh, "purchase_kwh": solution.purchase * instance.unit_kwh,
                         "sale_kwh": solution.sale * instance.unit_kwh, "soc_kwh": solution.soc * instance.unit_kwh,
                         "buy_price": instance.buy_price, "sell_price": instance.sell_price})


def run_case_study(windows, config, quick=False):
    """Optimize heterogeneous prosumers independently and aggregate their behavior."""
    case, output = config["case_study"], Path(config["paths"]["results"])
    output.mkdir(parents=True, exist_ok=True)
    assets = _assets(windows, config)
    price_sets = {name: _microgrid_prices(windows, assets, *bounds, case["buy_scarcity_elasticity"])
                  for name, bounds in case["price_scenarios"].items()}
    primary_scenario = case["primary_price_scenario"]
    scales = case["quick_capacity_scales"] if quick else case["capacity_scales"]
    rows, schedules, primary = [], [], []
    for window in windows:
        household, season = window.attrs["household_id"], window.attrs["season"]
        for price_scenario, prices in price_sets.items():
            buy_price, sell_price = prices[season]
            empty = _instance(window, 0, assets.loc[household, "pv_ratio"], case["unit_kwh"], sell_price,
                              f"{price_scenario}_no_storage", buy_price)
            no_storage = _no_storage(empty)
            check_solution(empty, no_storage)
            row = _metrics(empty, no_storage, NO_STORAGE, 0, no_storage)
            row["price_scenario"] = price_scenario
            rows.append(row)
            for scale in scales:
                instance = _instance(window, assets.loc[household, "capacity_ratio"] * scale, assets.loc[household, "pv_ratio"],
                                     case["unit_kwh"], sell_price, f"{price_scenario}_capacity_{scale:g}", buy_price)
                heuristic, flexible, block = _self_consumption(instance), solve_algorithm_1_with_sales(instance), solve_algorithm_2(instance)
                for policy, solution in [(SELF_CONSUMPTION, heuristic), (ALGORITHM_1, flexible), (ALGORITHM_2, block)]:
                    row = _metrics(instance, solution, policy, scale, no_storage, heuristic)
                    row["price_scenario"] = price_scenario
                    rows.append(row)
                if price_scenario == primary_scenario and scale == case["primary_capacity_scale"]:
                    if block.objective < flexible.objective - 1e-7:
                        raise AssertionError("block policy improves on its flexible relaxation")
                    primary.append(instance)
                    schedules += [_schedule(empty, no_storage, NO_STORAGE, 0, window.index, price_scenario),
                                  _schedule(instance, heuristic, SELF_CONSUMPTION, scale, window.index, price_scenario),
                                  _schedule(instance, flexible, ALGORITHM_1, scale, window.index, price_scenario),
                                  _schedule(instance, block, ALGORITHM_2, scale, window.index, price_scenario)]
    results, schedules = pd.DataFrame(rows), pd.concat(schedules, ignore_index=True)
    rng = np.random.default_rng(config["seed"] + 31)
    households = np.array(sorted(assets.index))
    rng.shuffle(households)
    participation = []
    for season in sorted(schedules.season.unique()):
        season_data = schedules[schedules.season.eq(season)]
        for rate in case["participation_rates"]:
            participating = set(households[: int(round(rate * len(households)))])
            chosen = pd.concat([season_data[(season_data.household_id.eq(household)) & season_data.policy.eq(ALGORITHM_1 if household in participating else SELF_CONSUMPTION)] for household in households])
            aggregate = chosen.groupby("timestamp")[["purchase_kwh", "sale_kwh"]].sum()
            bills = results[results.season.eq(season) & results.price_scenario.eq(primary_scenario) & results.capacity_scale.eq(case["primary_capacity_scale"])]
            bills = bills[((bills.household_id.isin(participating)) & bills.policy.eq(ALGORITHM_1)) | ((~bills.household_id.isin(participating)) & bills.policy.eq(SELF_CONSUMPTION))]
            participation.append({"season": season, "participation_rate": rate, "community_bill_gbp": bills.net_bill_gbp.sum(),
                                  "import_kwh": aggregate.purchase_kwh.sum(), "export_kwh": aggregate.sale_kwh.sum(),
                                  "peak_import_kw": 2 * aggregate.purchase_kwh.max(),
                                  "mean_absolute_ramp_kw": 2 * aggregate.purchase_kwh.diff().abs().mean()})
    participation = pd.DataFrame(participation)
    assets.reset_index().to_csv(output / "prosumer_assets.csv", index=False)
    results.to_csv(output / "case_study.csv", index=False)
    schedules.to_csv(output / "case_schedules.csv.gz", index=False, compression="gzip")
    participation.to_csv(output / "microgrid_participation.csv", index=False)
    return results, schedules, participation, primary


def _days(windows):
    blocks = []
    for window in windows:
        for start in range(0, len(window) - 47, 48):
            part = window.iloc[start : start + 48]
            blocks.append((part.demand_kwh.to_numpy(float), part.pv_kwh_per_kwp.to_numpy(float), part.buy_price.to_numpy(float)))
    return blocks


def _sample_days(blocks, n, rng):
    indices = rng.integers(0, len(blocks), int(np.ceil(n / 48)))
    return tuple(np.concatenate([blocks[i][column] for i in indices])[:n] for column in range(3))


def runtime_instance(blocks, n, capacity, seed, config):
    """Build an empirical multi-day instance with nondegenerate general limits."""
    rng, design = np.random.default_rng(seed), config["runtime"]
    demand_kwh, pv_shape, tariff = _sample_days(blocks, n, rng)
    demand_kwh *= rng.lognormal(0, 0.18)
    pv_ratio = float(design["pv_load_ratios"][seed % len(design["pv_load_ratios"])])
    pv_kwh = pv_shape * pv_ratio * demand_kwh.sum() / pv_shape.sum() if pv_shape.sum() else np.zeros(n)
    aggregate = np.zeros(n)
    for _ in range(design["microgrid_size"]):
        load, pv, _ = _sample_days(blocks, n, rng)
        load *= rng.lognormal(0, 0.28)
        if rng.random() < 0.65 and pv.sum():
            pv *= rng.lognormal(-0.1, 0.35) * load.sum() / pv.sum()
        else:
            pv[:] = 0
        aggregate += load - pv
    z = (aggregate - aggregate.mean()) / max(aggregate.std(), 1e-12)
    noise, ar = rng.standard_t(5, n) * 0.055, np.zeros(n)
    for period in range(n):
        ar[period] = 0.82 * (ar[period - 1] if period else 0) + np.sqrt(1 - 0.82**2) * noise[period]
    spikes = (rng.random(n) < 0.008) * rng.exponential(0.30, n)
    adjustment = 0.12 * z + ar + spikes
    buy_price = np.clip(tariff * np.exp(adjustment - np.median(adjustment)), 0.02, 1.50)
    sell_price = design["sell_ratio"] * buy_price
    unit = design["unit_kwh"]
    demand, pv = quantize(demand_kwh, unit), quantize(pv_kwh, unit)
    residual, excess = np.maximum(demand - pv, 0), np.maximum(pv - demand, 0)
    headroom = np.minimum(capacity, rng.gamma(2, max(1, capacity / 12), n).round().astype(int))
    regime = "load_only" if pv_ratio == 0 else "load_dominant" if pv_ratio < 1 else "balanced" if pv_ratio == 1 else "pv_dominant"
    instance = Instance(demand, pv, buy_price, sell_price, capacity, int(rng.integers(0, capacity + 1)), residual + headroom,
                        excess + capacity + headroom, unit, f"n{n}_C{capacity}_R{pv_ratio:g}_seed{seed}",
                        {"n": n, "capacity_units": capacity, "seed": seed, "pv_load_ratio": pv_ratio, "profile_regime": regime,
                         "demand_rounding_kwh": demand.sum() * unit - demand_kwh.sum(),
                         "pv_rounding_kwh": pv.sum() * unit - pv_kwh.sum()})
    if instance.demand.sum() == 0 or np.unique(instance.buy_price).size < 0.9 * n or np.mean(instance.buy_limit > 0) < 0.5:
        raise AssertionError("degenerate runtime instance")
    return instance


def _timed(solve, minimum_seconds, maximum_repeats=31):
    solution, samples = solve(), []
    while len(samples) < maximum_repeats and (len(samples) < 3 or sum(samples) < minimum_seconds):
        start = perf_counter_ns()
        current = solve()
        samples.append((perf_counter_ns() - start) / 1e9)
        if not np.isclose(current.objective, solution.objective, atol=1e-8, rtol=1e-9):
            raise AssertionError("nondeterministic algorithm objective")
    return solution, float(np.median(samples)), len(samples)


def benchmark(instance, config):
    rows, exact = [], {}
    for algorithm, block, solve in [(ALGORITHM_1, False, lambda: solve_algorithm_1_with_sales(instance)),
                                    (ALGORITHM_2, True, lambda: solve_algorithm_2(instance))]:
        solution, runtime, repetitions = _timed(solve, config["minimum_timing_seconds"])
        check_solution(instance, solution, block)
        exact[block] = solution.objective
        rows.append({**instance.metadata, "instance": instance.name, "algorithm": algorithm, "block": block,
                     "status": "optimal", "objective_gbp": solution.objective, "runtime_seconds": runtime,
                     "timing_repetitions": repetitions, "timing_method": "median warmed wall clock"})
    for block in (False, True):
        solution = solve_gurobi(instance, block, config["time_limit_seconds"], config["threads"])
        difference = solution.objective - exact[block] if solution.status == "optimal" else np.nan
        if solution.status == "optimal" and not np.isclose(difference, 0, atol=1e-7, rtol=1e-9):
            raise AssertionError(f"{instance.name}: solver difference {difference}")
        rows.append({**instance.metadata, "instance": instance.name, "algorithm": solution.algorithm, "block": block,
                     "status": solution.status, "objective_gbp": solution.objective,
                     "objective_difference": difference, "timing_method": "model build plus optimize wall clock", **solution.metadata})
    return pd.DataFrame(rows)


def run_runtime_benchmark(windows, config, quick=False):
    design, output = config["runtime"], Path(config["paths"]["results"]) / "runtime.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    setting = design["quick" if quick else "full"]
    previous = pd.read_csv(output) if output.exists() else pd.DataFrame()
    frames, blocks = [], _days(windows)
    expected = {ALGORITHM_1, ALGORITHM_2, "Gurobi LP", "Gurobi MIP"}
    warm = Instance([0], [0], [1.0], [0.0], 0, 0, [0], [0], name="gurobi_warmup")
    if solve_gurobi(warm, False, design["time_limit_seconds"], design["threads"]).status != "optimal":
        raise RuntimeError("Gurobi warm-up failed")
    for n in setting["horizons"]:
        for capacity in setting["capacities"]:
            for seed in setting["seeds"]:
                ratio = design["pv_load_ratios"][seed % len(design["pv_load_ratios"])]
                name = f"n{n}_C{capacity}_R{ratio:g}_seed{seed}"
                prior = previous[previous.instance.eq(name)] if not previous.empty else pd.DataFrame()
                if not prior.empty and set(prior.algorithm) == expected and prior.status.isin(["optimal", "time_limit"]).all():
                    frames.append(prior)
                    continue
                frame = benchmark(runtime_instance(blocks, n, capacity, seed, config), design)
                frames.append(frame)
                pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output, index=False)
    return results


def _bounded_ar_errors(n, epsilon, rng, rho=0.8):
    innovations, values = rng.normal(size=n), np.zeros(n)
    for period in range(n):
        values[period] = rho * (values[period - 1] if period else 0) + np.sqrt(1 - rho**2) * innovations[period]
    return epsilon * np.tanh(values)


def run_robustness(instances, config, quick=False):
    """Price-forecast sensitivity without sales and exploratory sensitivity with sales."""
    setting = config["robustness"]["quick" if quick else "full"]
    loads = {x.metadata["household_id"]: x.metadata["annual_load_kwh"] for x in instances}
    households = sorted(loads, key=loads.get)
    positions = np.linspace(0, len(households) - 1, min(setting["households"], len(households))).round().astype(int)
    selected = {households[i] for i in positions}
    cases = [x for x in instances if x.metadata["household_id"] in selected]
    rng, rows = np.random.default_rng(config["seed"] + 47), []
    for empirical in cases:
        no_sales = Instance(empirical.demand, np.zeros(empirical.n, dtype=int), empirical.buy_price, np.zeros(empirical.n),
                            empirical.capacity, empirical.initial_soc, empirical.demand + empirical.capacity,
                            np.zeros(empirical.n, dtype=int), empirical.unit_kwh, empirical.name + "_purchase_only", empirical.metadata)
        paths = [(_bounded_ar_errors(empirical.n, 1, rng), _bounded_ar_errors(empirical.n, 1, rng))
                 for _ in range(setting["replications"])]
        for scenario, sales_allowed, instance, solve in [("purchase_only", False, no_sales, solve_algorithm_1),
                                                         ("sales_enabled", True, empirical, solve_algorithm_1_with_sales)]:
            optimum = solve(instance)
            gross_import = instance.unit_kwh * float(np.dot(instance.buy_price, np.maximum(instance.demand - instance.pv, 0)))
            for replication, (base_buy, base_sell) in enumerate(paths):
                for epsilon in setting["epsilons"]:
                    buy_error = epsilon * base_buy
                    predicted_buy = instance.buy_price * (1 + buy_error)
                    if sales_allowed:
                        sell_error = epsilon * np.clip(0.7 * base_buy + np.sqrt(1 - 0.7**2) * base_sell, -1, 1)
                        predicted_sell = np.minimum(instance.sell_price * (1 + sell_error), predicted_buy)
                    else:
                        predicted_sell = np.zeros(instance.n)
                    predicted = replace(instance, buy_price=predicted_buy, sell_price=predicted_sell)
                    schedule = solve(predicted)
                    realized = instance.unit_kwh * float(np.dot(instance.buy_price, schedule.purchase) - np.dot(instance.sell_price, schedule.sale))
                    regret = max(0.0, realized - optimum.objective)
                    rows.append({"instance": instance.name, "scenario": scenario, "household_id": instance.metadata["household_id"], "season": instance.metadata["season"],
                                 "sales_allowed": sales_allowed, "epsilon": epsilon, "replication": replication,
                                 "true_optimum_gbp": optimum.objective, "predicted_optimum_gbp": schedule.objective,
                                 "realized_cost_gbp": realized, "regret_gbp": regret,
                                 "normalized_regret_percent": 100 * regret / gross_import if gross_import else np.nan,
                                 "schedule_change_kwh": instance.unit_kwh * (np.abs(schedule.purchase - optimum.purchase).sum() + np.abs(schedule.sale - optimum.sale).sum()),
                                 "realized_ratio": realized / optimum.objective if not sales_allowed and optimum.objective > 0 else np.nan,
                                 "guarantee": (1 + epsilon) / (1 - epsilon) if not sales_allowed else np.nan,
                                 "negative_true_bill": optimum.objective < 0})
    results = pd.DataFrame(rows)
    Path(config["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(config["paths"]["results"]) / "robustness.csv", index=False)
    return results
