"""Compact paper figures for the three experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from .experiments import ALGORITHM_1, ALGORITHM_2


COLORS = {"custom": "#173F5F", "gurobi": "#ED553B", "pv": "#F6C85F", "sale": "#3CAEA3", "gray": "#6B7280"}


def style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.22, "legend.fontsize": 8, "figure.constrained_layout.use": True})


def save(fig, name, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", bbox_inches="tight")


def _line(ax, data, x, algorithm):
    group = data[data.algorithm.eq(algorithm)].groupby(x).runtime_seconds
    summary = group.quantile([0.25, 0.5, 0.75]).unstack()
    color = COLORS["gurobi"] if algorithm.startswith("Gurobi") else COLORS["custom"]
    ax.plot(summary.index, summary[0.5], marker="o", ms=3, color=color, label=algorithm)
    ax.fill_between(summary.index.to_numpy(float), summary[0.25].to_numpy(float), summary[0.75].to_numpy(float), color=color, alpha=0.14)
    timed_out = data[data.algorithm.eq(algorithm) & data.status.eq("time_limit")].groupby(x).runtime_seconds.median()
    if not timed_out.empty:
        ax.scatter(timed_out.index, timed_out, marker="^", facecolors="none", edgecolors=color, zorder=4)


def runtime_scaling(results):
    style()
    valid = results.query("status in ['optimal', 'time_limit'] and runtime_seconds > 0")
    capacities, horizons = np.sort(valid.capacity_units.unique()), np.sort(valid.n.unique())
    fixed_capacity, fixed_horizon = capacities[len(capacities) // 2], horizons[len(horizons) // 2]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
    for column, (block, names, title) in enumerate([(False, [ALGORITHM_1, "Gurobi LP"], "Flexible purchases"),
                                                    (True, [ALGORITHM_2, "Gurobi MIP"], "Block purchases")]):
        horizon_data = valid[valid.block.eq(block) & valid.capacity_units.eq(fixed_capacity)]
        capacity_data = valid[valid.block.eq(block) & valid.n.eq(fixed_horizon)]
        for name in names:
            _line(axes[0, column], horizon_data, "n", name)
            _line(axes[1, column], capacity_data, "capacity_units", name)
        axes[0, column].set_title(f"{title}: $C={fixed_capacity:g}$")
        axes[1, column].set_title(f"{title}: $n={fixed_horizon:g}$")
        axes[0, column].set_xlabel("Periods $n$")
        axes[1, column].set_xlabel("Capacity states $C$")
        for ax in axes[:, column]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(frameon=False)
        axes[0, column].set_xticks(horizons, [f"{value:g}" for value in horizons])
        axes[1, column].set_xticks(capacities, [f"{value:g}" for value in capacities])
        for ax in axes[:, column]:
            ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel("Runtime (seconds)")
    axes[1, 0].set_ylabel("Runtime (seconds)")
    return fig


def performance_profiles(results):
    style()
    valid = results.query("status in ['optimal', 'time_limit'] and runtime_seconds > 0")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (block, title) in zip(axes, [(False, "Flexible purchases"), (True, "Block purchases")]):
        subset = valid[valid.block.eq(block)]
        table = subset.pivot(index="instance", columns="algorithm", values="runtime_seconds").dropna()
        solved = subset.assign(solved=subset.status.eq("optimal")).pivot(index="instance", columns="algorithm", values="solved").loc[table.index]
        best = table.where(solved).min(axis=1)
        ratios = table.div(best, axis=0).where(solved, np.inf)
        maximum = ratios.replace(np.inf, np.nan).max().max()
        tau = np.geomspace(1, max(2, maximum), 200)
        for algorithm in ratios:
            values = ratios[algorithm].to_numpy()
            color = COLORS["gurobi"] if algorithm.startswith("Gurobi") else COLORS["custom"]
            ax.step(tau, [(values <= x).mean() for x in tau], where="post", label=algorithm, color=color)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"Performance ratio $\tau$")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Fraction of paired instances")
    return fig


def microgrid_outcomes(schedules, participation):
    style()
    seasons = sorted(schedules.season.unique())
    season = seasons[len(seasons) // 2]
    data = schedules[schedules.season.eq(season) & schedules.policy.eq(ALGORITHM_1)]
    aggregate = data.groupby("timestamp")[["demand_kwh", "pv_kwh", "purchase_kwh", "sale_kwh", "buy_price", "sell_price"]].agg({
        "demand_kwh": "sum", "pv_kwh": "sum", "purchase_kwh": "sum", "sale_kwh": "sum", "buy_price": "first", "sell_price": "first"})
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    flow, price, participation_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    flow.plot(aggregate.index, aggregate.demand_kwh, color=COLORS["custom"], label="Demand")
    flow.plot(aggregate.index, aggregate.pv_kwh, color=COLORS["pv"], label="PV")
    flow.fill_between(aggregate.index, aggregate.purchase_kwh, color="#20639B", alpha=0.3, label="Import")
    flow.fill_between(aggregate.index, -aggregate.sale_kwh, color=COLORS["sale"], alpha=0.3, label="Export")
    flow.set_ylabel("Microgrid energy (kWh/30 min)")
    flow.legend(frameon=False, ncol=4)
    price.step(aggregate.index, 100 * aggregate.buy_price, where="post", color=COLORS["gurobi"], label="Buy")
    price.step(aggregate.index, 100 * aggregate.sell_price, where="post", color=COLORS["sale"], label="Sell")
    price.set_ylabel("Price (p/kWh)")
    price.set_xlabel(f"Date (week beginning {season})")
    price.legend(frameon=False)
    for axis in (flow, price):
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    flow.tick_params(labelbottom=False)
    summary = participation.groupby("participation_rate").agg({"community_bill_gbp": "mean", "peak_import_kw": "mean"})
    baseline = summary.community_bill_gbp.iloc[0]
    participation_ax.plot(100 * summary.index, baseline - summary.community_bill_gbp, marker="o", color=COLORS["custom"], label="Bill savings")
    participation_ax.set_xlabel("Optimizing prosumers (%)")
    participation_ax.set_ylabel("Mean weekly community savings (GBP)")
    second = participation_ax.twinx()
    second.plot(100 * summary.index, summary.peak_import_kw, marker="s", color=COLORS["gurobi"], label="Peak import")
    second.set_ylabel("Mean peak import (kW)")
    lines = participation_ax.lines + second.lines
    participation_ax.legend(lines, [line.get_label() for line in lines], frameon=False)
    return fig


def prosumer_behavior(results, primary_scale=1.0, price_scenario="central"):
    style()
    data = results[results.price_scenario.eq(price_scenario) & results.capacity_scale.eq(primary_scale) & results.policy.eq(ALGORITHM_1)]
    seasons = sorted(data.season.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    values = [data[data.season.eq(season)].benefit_vs_self_consumption_percent.dropna() for season in seasons]
    axes[0].boxplot(values, patch_artist=True, showfliers=False,
                    boxprops={"facecolor": COLORS["custom"], "alpha": 0.65}, medianprops={"color": "black"})
    axes[0].set_xticks(np.arange(1, len(seasons) + 1), [pd.Timestamp(season).strftime("%b %d") for season in seasons])
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_xlabel("Season start")
    axes[0].set_ylabel("Incremental benefit / gross import cost (%)")
    flex = results[results.price_scenario.eq(price_scenario) & results.policy.eq(ALGORITHM_1)].copy()
    flex["group"] = np.where(flex.pv_ratio.gt(0), "PV owners", "Without PV")
    for group, color in [("PV owners", COLORS["custom"]), ("Without PV", COLORS["gray"])]:
        summary = flex[flex.group.eq(group)].groupby("capacity_scale").benefit_vs_self_consumption_percent.quantile([0.25, 0.5, 0.75]).unstack()
        if not summary.empty:
            axes[1].plot(summary.index, summary[0.5], marker="o", color=color, label=group)
            axes[1].fill_between(summary.index, summary[0.25], summary[0.75], color=color, alpha=0.16)
    axes[1].set_xlabel("Battery-capacity scale")
    axes[1].set_ylabel("Incremental benefit / gross import cost (%)")
    axes[1].legend(frameon=False)
    return fig


def forecast_robustness(results):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    purchase = results[~results.sales_allowed]
    summary = purchase.groupby("epsilon").realized_ratio.quantile([0.05, 0.5, 0.95]).unstack()
    guarantee = purchase.groupby("epsilon").guarantee.first()
    axes[0].fill_between(summary.index, summary[0.05], summary[0.95], color=COLORS["custom"], alpha=0.2, label="5-95% band")
    axes[0].plot(summary.index, summary[0.5], color=COLORS["custom"], label="Median")
    axes[0].plot(guarantee.index, guarantee, color=COLORS["gurobi"], linestyle="--", label="Theoretical bound")
    axes[0].set_title("Purchase-only")
    ticks = np.array([0, 0.05, 0.10, 0.20, 0.30])
    labels = ["0", "5", "10", "20", "30"]
    axes[0].set_xticks(ticks, labels)
    axes[0].set_xlabel(r"Maximum price error $\varepsilon$ (%)")
    axes[0].set_ylabel("Realized cost / true optimum")
    axes[0].legend(frameon=False)
    sales = results[results.sales_allowed]
    summary = sales.groupby("epsilon").normalized_regret_percent.quantile([0.05, 0.5, 0.95]).unstack()
    axes[1].fill_between(summary.index, summary[0.05], summary[0.95], color=COLORS["sale"], alpha=0.25, label="5-95% band")
    axes[1].plot(summary.index, summary[0.5], color=COLORS["sale"], label="Median")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_title("Sales allowed")
    axes[1].set_xticks(ticks, labels)
    axes[1].set_xlabel(r"Maximum price error $\varepsilon$ (%)")
    axes[1].set_ylabel("Regret / gross import cost (%)")
    axes[1].legend(frameon=False)
    return fig
