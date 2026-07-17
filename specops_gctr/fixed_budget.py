"""Target-free, fixed-budget optimization traces.

The original manuscript pipeline measures calls to an oracle target.  This
module provides the complementary deployment protocol used by the optimizer
portfolio: every arm receives the same maximum number of objective evaluations
and returns its best-so-far quality at pre-registered checkpoints.  No target
value is exposed to an optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .optimization import _project_trust_region
from .qaoa import qaoa_expectation


class BudgetExhausted(RuntimeError):
    """Raised when an optimizer attempts to exceed its registered budget."""


@dataclass(frozen=True)
class FixedBudgetTrace:
    """Best-so-far trace and exact resource accounting for one optimizer arm."""

    checkpoints: tuple[int, ...]
    best_values: tuple[float, ...]
    best_params: tuple[tuple[float, ...], ...]
    evaluations_used: int
    objective_queries: int
    shots_used: int
    backend: str = "exact_statevector"

    def ratios(self, optimum: float) -> tuple[float, ...]:
        if optimum <= 0:
            return tuple(0.0 for _ in self.best_values)
        return tuple(float(value / optimum) for value in self.best_values)

    def aurc(self, optimum: float) -> float:
        """Equal-weight area under the checkpointed normalized-regret curve."""
        ratios = np.asarray(self.ratios(optimum), dtype=float)
        return float(np.mean(1.0 - ratios))


class BudgetedObjective:
    """Exact QAOA evaluator with a hard call ceiling and a best-so-far trace."""

    def __init__(self, C: np.ndarray, n: int, budget: int):
        if int(budget) != budget or budget < 1:
            raise ValueError("budget must be a positive integer")
        self.C = np.asarray(C, dtype=float)
        self.n = int(n)
        self.budget = int(budget)
        self.count = 0
        self._best_value = -np.inf
        self._best_params: np.ndarray | None = None
        self._best_values: list[float] = []
        self._best_parameter_trace: list[np.ndarray] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.count

    def __call__(self, params: np.ndarray) -> float:
        if self.count >= self.budget:
            raise BudgetExhausted("fixed objective-evaluation budget exhausted")
        x = np.asarray(params, dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("parameters must be a finite one-dimensional vector")
        value = qaoa_expectation(self.C, x, self.n)
        self.count += 1
        if value > self._best_value:
            self._best_value = float(value)
            self._best_params = x.copy()
        assert self._best_params is not None
        self._best_values.append(self._best_value)
        self._best_parameter_trace.append(self._best_params.copy())
        return float(value)

    def finish(self, checkpoints: Iterable[int]) -> FixedBudgetTrace:
        points = tuple(int(point) for point in checkpoints)
        if not points or points != tuple(sorted(set(points))):
            raise ValueError("checkpoints must be non-empty, unique and increasing")
        if points[0] < 1 or points[-1] > self.budget:
            raise ValueError("checkpoints must lie within the registered budget")
        if self.count != self.budget:
            raise RuntimeError(
                f"optimizer used {self.count} evaluations, expected exactly "
                f"{self.budget}")
        values = tuple(float(self._best_values[point - 1]) for point in points)
        params = tuple(
            tuple(float(x) for x in self._best_parameter_trace[point - 1])
            for point in points
        )
        return FixedBudgetTrace(
            checkpoints=points,
            best_values=values,
            best_params=params,
            evaluations_used=self.count,
            objective_queries=self.count,
            shots_used=0,
        )


def _validated_scale(x0: np.ndarray, step_scale: np.ndarray | None) -> np.ndarray:
    if step_scale is None:
        return np.ones_like(x0)
    scale = np.asarray(step_scale, dtype=float)
    if (scale.shape != x0.shape or not np.all(np.isfinite(scale))
            or np.any(scale < 0) or not np.any(scale > 0)):
        raise ValueError("step_scale must be finite, nonnegative and match x0")
    return scale


def coordinate_trace(
    objective: BudgetedObjective,
    x0: np.ndarray,
    checkpoints: Iterable[int],
    *,
    step0: float = 0.3,
    step_scale: np.ndarray | None = None,
    candidates: Iterable[np.ndarray] | None = None,
    center: np.ndarray | None = None,
    L_inv: np.ndarray | None = None,
    radius: float | None = None,
    recenter_on_best: bool = False,
    rng: np.random.Generator | None = None,
) -> FixedBudgetTrace:
    """Run deterministic pattern search until the exact budget is consumed.

    When the local step falls below ``1e-3``, a small random restart is used so
    that every arm exposes a full anytime curve rather than silently receiving
    a smaller resource budget after numerical convergence.
    """
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    if not np.isfinite(step0) or step0 <= 0:
        raise ValueError("step0 must be positive")
    scale = _validated_scale(x, step_scale)
    rng = rng or np.random.default_rng(0)
    ctr = None if center is None else np.asarray(center, dtype=float).copy()

    seeds: list[np.ndarray] = []
    for seed in [x, *(list(candidates or []))]:
        candidate = np.asarray(seed, dtype=float)
        if candidate.shape != x.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("candidate seeds must be finite and match x0")
        if not any(np.array_equal(candidate, prior) for prior in seeds):
            seeds.append(candidate.copy())

    best_value = -np.inf
    best = x.copy()
    for seed in seeds:
        if objective.remaining == 0:
            break
        evaluated = seed
        if ctr is not None and not recenter_on_best:
            evaluated = _project_trust_region(seed, ctr, L_inv, radius)
        value = objective(evaluated)
        if value > best_value:
            best_value, best = value, evaluated.copy()
    x = best.copy()
    if recenter_on_best and ctr is not None:
        ctr = x.copy()

    step = float(step0)
    while objective.remaining:
        improved = False
        for dim in range(x.size):
            for sign in (1.0, -1.0):
                if objective.remaining == 0:
                    break
                proposal = x.copy()
                proposal[dim] += sign * step * scale[dim]
                if ctr is not None:
                    proposal = _project_trust_region(
                        proposal, ctr, L_inv, radius)
                value = objective(proposal)
                if value > best_value:
                    best_value, x = value, proposal.copy()
                    improved = True
        if not improved:
            step *= 0.5
        if step < 1e-3 and objective.remaining:
            # A target-free fixed-budget run should not stop because one local
            # basin converged. Restart near the incumbent, respecting geometry.
            proposal = x + rng.normal(scale=step0, size=x.size) * scale
            if ctr is not None:
                proposal = _project_trust_region(proposal, ctr, L_inv, radius)
            value = objective(proposal)
            if value > best_value:
                best_value, x = value, proposal.copy()
            step = float(step0)
    return objective.finish(checkpoints)


def random_multistart_trace(
    objective: BudgetedObjective,
    dimension: int,
    checkpoints: Iterable[int],
    *,
    restarts: int = 4,
    rng: np.random.Generator | None = None,
) -> FixedBudgetTrace:
    """Random-start pattern search with equal per-restart sub-budgets."""
    if int(dimension) != dimension or dimension < 2 or dimension % 2:
        raise ValueError("dimension must be a positive gamma/beta pair count")
    if int(restarts) != restarts or restarts < 1:
        raise ValueError("restarts must be positive")
    rng = rng or np.random.default_rng(0)
    restarts = min(int(restarts), objective.budget)
    boundaries = np.linspace(0, objective.budget, restarts + 1, dtype=int)[1:]
    global_best = -np.inf
    global_x: np.ndarray | None = None
    for boundary in boundaries:
        x = np.empty(int(dimension), dtype=float)
        x[0::2] = rng.uniform(0.0, np.pi, size=x[0::2].size)
        x[1::2] = rng.uniform(0.0, np.pi / 2.0, size=x[1::2].size)
        value = objective(x)
        local_best = value
        if value > global_best:
            global_best, global_x = value, x.copy()
        step = 0.3
        while objective.count < boundary:
            improved = False
            for dim in range(x.size):
                for sign in (1.0, -1.0):
                    if objective.count >= boundary:
                        break
                    proposal = x.copy()
                    proposal[dim] += sign * step
                    value = objective(proposal)
                    if value > global_best:
                        global_best, global_x = value, proposal.copy()
                    if value > local_best:
                        local_best = value
                        x = proposal
                        improved = True
            if not improved:
                step *= 0.5
            if step < 1e-3:
                step = 0.3
    assert global_x is not None
    return objective.finish(checkpoints)


def spsa_trace(
    objective: BudgetedObjective,
    x0: np.ndarray,
    checkpoints: Iterable[int],
    *,
    rng: np.random.Generator | None = None,
    a: float = 0.18,
    c: float = 0.12,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> FixedBudgetTrace:
    """Maximizing SPSA with every plus/minus evaluation retained in the trace."""
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    rng = rng or np.random.default_rng(0)
    iteration = 0
    while objective.remaining:
        if objective.remaining == 1:
            objective(x)
            break
        ak = a / ((iteration + 1) ** alpha)
        ck = c / ((iteration + 1) ** gamma)
        delta = rng.choice(np.array([-1.0, 1.0]), size=x.size)
        plus = x + ck * delta
        minus = x - ck * delta
        f_plus = objective(plus)
        f_minus = objective(minus)
        gradient = ((f_plus - f_minus) / (2.0 * ck)) * delta
        x = x + ak * gradient
        iteration += 1
    return objective.finish(checkpoints)


def run_exact_trace(
    C: np.ndarray,
    n: int,
    budget: int,
    checkpoints: Iterable[int],
    runner: Callable[[BudgetedObjective], FixedBudgetTrace],
) -> FixedBudgetTrace:
    """Convenience wrapper used by portfolio arms and tests."""
    objective = BudgetedObjective(C, n, budget)
    return runner(objective)
