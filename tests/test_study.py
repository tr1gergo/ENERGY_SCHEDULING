from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microgrid.core import InfeasibleError, Instance, check_solution, quantize, solve_algorithm_1, solve_algorithm_1_with_sales, solve_algorithm_2, solve_gurobi
from microgrid.experiments import _days, _instance, _self_consumption, run_robustness, runtime_instance


def oracle(instance, block):
    values = {instance.initial_soc: 0.0}
    for period in range(instance.n):
        updated = {}
        purchases = [0, int(instance.buy_limit[period])] if block else range(int(instance.buy_limit[period]) + 1)
        for prior, cost in values.items():
            for purchase in purchases:
                for sale in range(int(instance.sell_limit[period]) + 1):
                    soc = prior + int(instance.pv[period] - instance.demand[period]) + purchase - sale
                    if 0 <= soc <= instance.capacity:
                        objective = cost + instance.unit_kwh * (instance.buy_price[period] * purchase - instance.sell_price[period] * sale)
                        updated[soc] = min(updated.get(soc, np.inf), objective)
        values = updated
    if not values:
        raise InfeasibleError
    return min(values.values())


def random_instance(seed):
    rng, n, capacity = np.random.default_rng(seed), 5, 6
    demand, pv = rng.integers(0, 4, n), rng.integers(0, 4, n)
    buy = rng.uniform(0.05, 0.60, n)
    return Instance(demand, pv, buy, buy * rng.uniform(0, 1, n), capacity, int(rng.integers(capacity + 1)),
                    rng.integers(0, 7, n), rng.integers(0, 8, n), 0.1, f"random_{seed}")


@pytest.mark.parametrize("seed", range(80))
def test_algorithms_match_independent_state_transition_oracle(seed):
    instance = random_instance(seed)
    for block, solve in [(False, solve_algorithm_1_with_sales), (True, solve_algorithm_2)]:
        try:
            expected = oracle(instance, block)
        except InfeasibleError:
            with pytest.raises(InfeasibleError):
                solve(instance)
            continue
        solution = solve(instance)
        assert solution.objective == pytest.approx(expected, abs=1e-9)
        if block:
            assert solution.metadata["complexity"] == "O(nC)"
        check_solution(instance, solution, block)

    purchase_only = replace(instance, sell_price=np.zeros(instance.n), sell_limit=np.zeros(instance.n, dtype=int))
    try:
        expected = oracle(purchase_only, False)
    except InfeasibleError:
        with pytest.raises(InfeasibleError):
            solve_algorithm_1(purchase_only)
    else:
        assert solve_algorithm_1(purchase_only).objective == pytest.approx(expected, abs=1e-9)


def test_cumulative_rounding_preserves_trace_energy():
    observations = np.full(336, 0.04)
    units = quantize(observations, 0.1)
    assert units.sum() > 0
    assert abs(units.sum() * 0.1 - observations.sum()) <= 0.05 + 1e-12


def window(season="2013-07-15"):
    index = pd.date_range(season, periods=96, freq="30min")
    hour = np.arange(96) % 48 / 2
    frame = pd.DataFrame({"demand_kwh": 0.25 + 0.15 * ((hour >= 17) & (hour <= 22)),
                          "pv_kwh_per_kwp": np.maximum(0, np.sin((hour - 6) * np.pi / 14)) / 2,
                          "buy_price": np.where((hour >= 16) & (hour <= 20), 0.50, 0.12)}, index=index)
    frame.attrs = {"household_id": "H1", "season": season, "annual_load_kwh": 2600.0, "annual_pv_yield": 950.0}
    return frame


def test_assets_are_fixed_and_runtime_instance_is_nondegenerate():
    first, second = window(), window("2013-10-14")
    sell = np.full(96, 0.05)
    a = _instance(first, 1.0, 0.8, 0.1, sell)
    b = _instance(second, 1.0, 0.8, 0.1, sell)
    assert a.capacity == b.capacity
    assert a.metadata["pv_kwp"] == pytest.approx(b.metadata["pv_kwp"])
    config = {"runtime": {"microgrid_size": 4, "sell_ratio": 0.5, "unit_kwh": 0.1, "pv_load_ratios": [0.0, 0.5, 1.0, 1.5]}}
    generated = runtime_instance(_days([first]), 48, 10, 9, config)
    assert generated.demand.sum() > 0
    assert np.unique(generated.buy_price).size >= 44
    assert generated.metadata["profile_regime"] == "load_dominant"
    assert solve_algorithm_2(generated).objective >= solve_algorithm_1_with_sales(generated).objective - 1e-9
    heuristic = _self_consumption(a)
    assert heuristic.objective >= solve_algorithm_1_with_sales(a).objective - 1e-9


def test_gurobi_exactness_and_zero_error_robustness(tmp_path):
    instance = _instance(window(), 0.75, 0.8, 0.1, np.full(96, 0.06))
    for block, solve in [(False, solve_algorithm_1_with_sales), (True, solve_algorithm_2)]:
        gurobi = solve_gurobi(instance, block, time_limit=30)
        assert gurobi.status == "optimal"
        assert gurobi.metadata["mip_gap"] == 0
        assert gurobi.metadata["runtime_seconds"] >= gurobi.metadata["optimization_seconds"]
        assert gurobi.objective == pytest.approx(solve(instance).objective, abs=1e-7)
    config = {"seed": 3, "paths": {"results": str(tmp_path)},
              "robustness": {"quick": {"households": 1, "epsilons": [0.0], "replications": 2}}}
    results = run_robustness([instance], config, quick=True)
    assert results.regret_gbp.max() < 1e-9
    assert set(results.sales_allowed) == {False, True}
    assert set(results.scenario) == {"purchase_only", "sales_enabled"}
