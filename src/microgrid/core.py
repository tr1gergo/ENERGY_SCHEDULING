"""Paper formulations, exact algorithms, validation, and Gurobi comparators."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


NEG_INF = float("-inf")


class InfeasibleError(RuntimeError):
    pass


def quantize(values_kwh, unit_kwh: float) -> np.ndarray:
    """Cumulative rounding keeps total energy error within half a unit."""
    values = np.asarray(values_kwh, dtype=float)
    if unit_kwh <= 0 or values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("energy must be a finite nonnegative vector and unit_kwh positive")
    cumulative = np.rint(np.cumsum(values / unit_kwh)).astype(np.int64)
    units = np.diff(np.r_[np.int64(0), cumulative])
    if np.any(units < 0):
        raise AssertionError("quantization produced a negative increment")
    return units


def _ints(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    rounded = np.rint(raw)
    if raw.ndim != 1 or np.any(~np.isfinite(raw)) or not np.allclose(raw, rounded) or np.any(rounded < 0):
        raise ValueError(f"{name} must be a finite nonnegative integer vector")
    return rounded.astype(np.int64)


def _floats(values, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or np.any(~np.isfinite(out)) or np.any(out < 0):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    return out


@dataclass(frozen=True)
class Instance:
    demand: Any
    pv: Any
    buy_price: Any
    sell_price: Any
    capacity: int
    initial_soc: int
    buy_limit: Any
    sell_limit: Any
    unit_kwh: float = 1.0
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        vectors = {
            "demand": _ints(self.demand, "demand"),
            "pv": _ints(self.pv, "pv"),
            "buy_price": _floats(self.buy_price, "buy_price"),
            "sell_price": _floats(self.sell_price, "sell_price"),
            "buy_limit": _ints(self.buy_limit, "buy_limit"),
            "sell_limit": _ints(self.sell_limit, "sell_limit"),
        }
        n = len(vectors["demand"])
        if n == 0 or any(len(value) != n for value in vectors.values()):
            raise ValueError("all instance vectors must have the same positive length")
        capacity, initial = int(self.capacity), int(self.initial_soc)
        if capacity < 0 or not 0 <= initial <= capacity or self.unit_kwh <= 0:
            raise ValueError("require unit_kwh > 0 and 0 <= initial_soc <= capacity")
        if np.any(vectors["sell_price"] > vectors["buy_price"] + 1e-12):
            raise ValueError("the paper assumes sell_price <= buy_price in every period")
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "initial_soc", initial)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n(self):
        return len(self.demand)


@dataclass
class Solution:
    objective: float
    purchase: np.ndarray | None
    sale: np.ndarray | None
    soc: np.ndarray | None
    status: str = "optimal"
    algorithm: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def check_solution(instance: Instance, solution: Solution, block: bool = False, atol: float = 1e-7):
    if solution.status != "optimal" or any(x is None for x in (solution.purchase, solution.sale, solution.soc)):
        raise AssertionError("an optimal reconstructed solution is required")
    purchase, sale, soc = map(lambda x: np.asarray(x, dtype=float), (solution.purchase, solution.sale, solution.soc))
    if any(len(x) != instance.n for x in (purchase, sale, soc)):
        raise AssertionError("schedule length mismatch")
    if np.any(purchase < -atol) or np.any(purchase > instance.buy_limit + atol):
        raise AssertionError("purchase bound violation")
    if np.any(sale < -atol) or np.any(sale > instance.sell_limit + atol):
        raise AssertionError("sales bound violation")
    if block and not np.all(np.isclose(purchase, 0, atol=atol) | np.isclose(purchase, instance.buy_limit, atol=atol)):
        raise AssertionError("block-purchase violation")
    expected_soc = instance.initial_soc + np.cumsum(instance.pv - instance.demand + purchase - sale)
    if not np.allclose(soc, expected_soc, atol=atol) or np.any(soc < -atol) or np.any(soc > instance.capacity + atol):
        raise AssertionError("battery-balance violation")
    objective = instance.unit_kwh * (np.dot(instance.buy_price, purchase) - np.dot(instance.sell_price, sale))
    if not np.isclose(solution.objective, objective, atol=atol, rtol=1e-9):
        raise AssertionError("objective mismatch")


def _soc(instance: Instance, purchase: np.ndarray, sale: np.ndarray) -> np.ndarray:
    return (instance.initial_soc + np.cumsum(instance.pv - instance.demand + purchase - sale)).astype(float)


def _purchase_only_feasible(instance: Instance) -> bool:
    low = high = instance.initial_soc
    for flow, alpha in zip(instance.pv - instance.demand, instance.buy_limit):
        low = max(0, low + int(flow))
        high = min(instance.capacity, high + int(flow) + int(alpha))
        if low > high:
            return False
    return True


def solve_algorithm_1(instance: Instance) -> Solution:
    """Algorithm 1 (Buying backwards) for PEAC-B-General in O(n^2)."""
    if np.any(instance.sell_limit):
        raise ValueError("Algorithm 1 expects a purchase-only instance")
    if not _purchase_only_feasible(instance):
        raise InfeasibleError(instance.name)
    n = instance.n
    price = np.r_[0.0, instance.buy_price]
    alpha = np.r_[0, instance.buy_limit].astype(np.int64)
    residual = np.r_[0, np.maximum(instance.demand - instance.pv, 0)].astype(np.int64)
    excess = np.r_[0, np.maximum(instance.pv - instance.demand, 0)].astype(np.int64)
    purchase = np.zeros(n + 1, dtype=np.int64)
    soc = np.zeros(n + 1, dtype=np.int64)
    remaining = instance.capacity + residual.copy()
    soc[0], remaining[0] = instance.initial_soc, instance.capacity - instance.initial_soc
    latest_full, candidates = 0, []

    def cheapest(period):
        while candidates:
            _, _, index = candidates[0]
            if index <= latest_full or index > period or purchase[index] >= alpha[index] or remaining[index] <= 0:
                heapq.heappop(candidates)
            else:
                return index
        raise InfeasibleError(instance.name)

    for period in range(1, n + 1):
        if alpha[period] > 0:
            heapq.heappush(candidates, (float(price[period]), -period, period))
        if excess[period] > 0:
            soc[period] = soc[period - 1] + excess[period]
            remaining[period] = remaining[period - 1] - excess[period]
            if latest_full + 1 < period:
                remaining[latest_full + 1 : period] = np.minimum(remaining[latest_full + 1 : period], remaining[period])
        elif soc[period - 1] >= residual[period]:
            soc[period] = soc[period - 1] - residual[period]
            remaining[period] = remaining[period - 1] + residual[period]
        else:
            unmet = int(residual[period] - soc[period - 1])
            while unmet:
                index = cheapest(period)
                increment = min(int(remaining[index]), int(alpha[index] - purchase[index]), unmet)
                purchase[index] += increment
                if index < period:
                    soc[index:period] += increment
                    remaining[index:period] -= increment
                    zeros = np.flatnonzero(remaining[index:period] == 0)
                    if len(zeros):
                        latest_full = max(latest_full, index + int(zeros[-1]))
                if latest_full + 1 < index:
                    remaining[latest_full + 1 : index] = np.minimum(remaining[latest_full + 1 : index], remaining[index])
                unmet -= increment
            soc[period], remaining[period] = 0, instance.capacity
        if soc[period] > instance.capacity:
            raise InfeasibleError(instance.name)
        if soc[period] == instance.capacity:
            latest_full = period
    x = purchase[1:].astype(float)
    solution = Solution(instance.unit_kwh * float(np.dot(instance.buy_price, x)), x, np.zeros(n), _soc(instance, x, np.zeros(n)), algorithm="Algorithm 1", metadata={"complexity": "O(n^2)"})
    check_solution(instance, solution)
    return solution


def solve_algorithm_1_with_sales(instance: Instance) -> Solution:
    """Theorem 5 reduction followed by Algorithm 1 for PEAC-BS-General."""
    residual = np.maximum(instance.demand - instance.pv, 0)
    excess = np.maximum(instance.pv - instance.demand, 0)
    transformed = Instance(
        demand=np.column_stack((residual, instance.sell_limit)).ravel(),
        pv=np.column_stack((np.zeros(instance.n, dtype=int), excess)).ravel(),
        buy_price=np.column_stack((instance.buy_price, instance.sell_price)).ravel(),
        sell_price=np.zeros(2 * instance.n),
        capacity=instance.capacity,
        initial_soc=instance.initial_soc,
        buy_limit=np.column_stack((instance.buy_limit, instance.sell_limit)).ravel(),
        sell_limit=np.zeros(2 * instance.n, dtype=int),
        unit_kwh=instance.unit_kwh,
        name=instance.name,
    )
    reduced = solve_algorithm_1(transformed)
    purchase = reduced.purchase[0::2].copy()
    sale = instance.sell_limit.astype(float) - reduced.purchase[1::2]
    solution = Solution(
        instance.unit_kwh * float(np.dot(instance.buy_price, purchase) - np.dot(instance.sell_price, sale)),
        purchase, sale, _soc(instance, purchase, sale), algorithm="Algorithm 1 + Theorem 5", metadata={"complexity": "O(n^2)"},
    )
    check_solution(instance, solution)
    return solution


def _moving_max(values, offset, width):
    """Maxima and argmaxes of monotonically shifting integer intervals in O(C)."""
    capacity, added = len(values) - 1, -1
    maxima = np.full_like(values, NEG_INF)
    indices = np.full(len(values), -1, dtype=np.int32)
    candidates = deque()
    for state in range(capacity + 1):
        lower, upper = max(0, state + offset), min(capacity, state + offset + width)
        while added < upper:
            added += 1
            if np.isfinite(values[added]):
                while candidates and values[candidates[-1]] < values[added]:
                    candidates.pop()
                candidates.append(added)
        while candidates and candidates[0] < lower:
            candidates.popleft()
        if lower <= upper and candidates:
            indices[state] = candidates[0]
            maxima[state] = values[candidates[0]]
    return maxima, indices


def solve_algorithm_2(instance: Instance, reconstruct: bool = True) -> Solution:
    """Algorithm 2 for PEAC-BS-Ex-General in O(nC) using monotone deques."""
    C = instance.capacity
    values = np.full(C + 1, NEG_INF)
    values[instance.initial_soc] = 0.0
    states = np.arange(C + 1, dtype=float)
    parents = np.full((instance.n, C + 1), -1, dtype=np.int32) if reconstruct else None
    choices = np.zeros((instance.n, C + 1), dtype=np.int8) if reconstruct else None
    for period in range(instance.n):
        alpha, beta = int(instance.buy_limit[period]), int(instance.sell_limit[period])
        net, buy, sell = int(instance.demand[period] - instance.pv[period]), float(instance.buy_price[period]), float(instance.sell_price[period])
        weighted, updated = values + sell * states, np.full(C + 1, NEG_INF)
        for choice, purchase in ((0, 0), (1, alpha)):
            maxima, previous = _moving_max(weighted, net - purchase, beta)
            candidate = maxima + sell * (purchase - net - states) - buy * purchase
            improve = (previous >= 0) & (candidate > updated)
            updated[improve] = candidate[improve]
            if reconstruct:
                parents[period, improve], choices[period, improve] = previous[improve], choice
        values = updated
    if not np.any(np.isfinite(values)):
        raise InfeasibleError(instance.name)
    final_soc = int(np.nanargmax(values))
    objective = -float(values[final_soc]) * instance.unit_kwh
    if not reconstruct:
        return Solution(objective, None, None, None, algorithm="Algorithm 2", metadata={"complexity": "O(nC)"})
    purchase, sale, current = np.zeros(instance.n), np.zeros(instance.n), final_soc
    for period in range(instance.n - 1, -1, -1):
        previous = int(parents[period, current])
        if previous < 0:
            raise RuntimeError("broken predecessor chain")
        if choices[period, current]:
            purchase[period] = instance.buy_limit[period]
        sale[period] = previous + purchase[period] + instance.pv[period] - instance.demand[period] - current
        current = previous
    solution = Solution(objective, purchase, sale, _soc(instance, purchase, sale), algorithm="Algorithm 2", metadata={"complexity": "O(nC)"})
    check_solution(instance, solution, block=True)
    return solution


def solve_gurobi(instance: Instance, block: bool, time_limit=180.0, threads=1) -> Solution:
    """Exact LP/MIP comparator with a closed proof gap."""
    import gurobipy as gp
    from gurobipy import GRB

    build_start = perf_counter()
    model = gp.Model(instance.name or "PEAC")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.Seed = 20260904
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.FeasibilityTol = model.Params.OptimalityTol = model.Params.IntFeasTol = 1e-9
    periods = range(instance.n)
    soc = model.addVars(periods, lb=0, ub=instance.capacity)
    sale = model.addVars(periods, lb=0, ub={p: int(instance.sell_limit[p]) for p in periods})
    if block:
        choose = model.addVars(periods, vtype=GRB.BINARY)
        purchase = {p: int(instance.buy_limit[p]) * choose[p] for p in periods}
    else:
        purchase_vars = model.addVars(periods, lb=0, ub={p: int(instance.buy_limit[p]) for p in periods})
        purchase = {p: purchase_vars[p] for p in periods}
    for p in periods:
        prior = instance.initial_soc if p == 0 else soc[p - 1]
        model.addConstr(soc[p] == prior + int(instance.pv[p] - instance.demand[p]) + purchase[p] - sale[p])
    model.setObjective(instance.unit_kwh * gp.quicksum(instance.buy_price[p] * purchase[p] - instance.sell_price[p] * sale[p] for p in periods), GRB.MINIMIZE)
    build_seconds = perf_counter() - build_start
    start = perf_counter()
    model.optimize()
    solve_seconds = perf_counter() - start
    names = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "inf_or_unbd"}
    status, incumbent = names.get(model.Status, str(model.Status)), model.SolCount > 0
    x = y = s = None
    objective = float("nan")
    if incumbent:
        x = np.array([instance.buy_limit[p] * choose[p].X if block else purchase_vars[p].X for p in periods])
        y, s, objective = np.array([sale[p].X for p in periods]), np.array([soc[p].X for p in periods]), float(model.ObjVal)
    solution = Solution(objective, x, y, s, status=status, algorithm="Gurobi MIP" if block else "Gurobi LP", metadata={
        "runtime_seconds": build_seconds + solve_seconds, "optimization_seconds": solve_seconds,
        "solver_runtime_seconds": float(model.Runtime), "build_seconds": build_seconds,
        "mip_gap": float(model.MIPGap) if block and incumbent else 0.0, "best_bound": float(model.ObjBound) if incumbent else np.nan,
        "nodes": float(model.NodeCount), "work": float(model.Work), "solutions": int(model.SolCount),
    })
    if status == "optimal":
        check_solution(instance, solution, block=block)
    return solution
