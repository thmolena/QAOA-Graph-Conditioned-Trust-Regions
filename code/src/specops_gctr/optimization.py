"""Query-counting local search and search policies.

The single resource we count is the number of *objective evaluations* (QAOA
expectation queries). This mirrors hardware, where each query means repeated
state preparation, execution and measurement.
"""
from __future__ import annotations

import numpy as np

from .qaoa import (qaoa_expectation, qaoa_expectation_sampled,
                   best_bitstring_ratio)


class QueryCounter:
    """Counts objective queries; optionally returns finite-shot estimates.

    With ``shots=None`` each call returns the exact statevector expectation
    (infinite-shot limit). With an integer ``shots`` each call returns the
    unbiased sample-mean estimator a hardware run would compute (mean cut value
    of `shots` Born-rule bitstrings), so the optimizer makes its decisions
    under realistic estimator noise. Either way one call is one counted query.
    """

    def __init__(self, C, n, maxcut, shots=None, noise_rng=None):
        self.C = C
        self.n = n
        self.maxcut = maxcut
        self.count = 0
        self.shots = shots
        self.noise_rng = noise_rng or np.random.default_rng(0)

    def __call__(self, params) -> float:
        self.count += 1
        params = np.asarray(params)
        if self.shots is None:
            return qaoa_expectation(self.C, params, self.n)
        return qaoa_expectation_sampled(self.C, params, self.n, self.shots,
                                        rng=self.noise_rng)


class CallableQueryCounter:
    """Count calls to a user-supplied objective evaluator.

    ``evaluator`` may execute a circuit, call a remote service, read a
    surrogate, or evaluate any other scalar objective. The search policy sees
    only the returned scalar and an exact call count.
    """

    def __init__(self, evaluator):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.evaluator = evaluator
        self.count = 0

    def __call__(self, params) -> float:
        self.count += 1
        value = float(self.evaluator(np.asarray(params, dtype=float)))
        if not np.isfinite(value):
            raise ValueError("objective evaluator returned a non-finite value")
        return value


def _project_trust_region(x, center, L_inv=None, radius=None):
    """Retract x into the Mahalanobis ball of `radius` around `center`.

    L_inv is a matrix such that ||L_inv (x-center)|| is the Mahalanobis distance
    (L_inv = Sigma^{-1/2}). Points outside the ball are pulled radially toward
    the center until the Mahalanobis norm equals the radius (a feasibility
    retraction along the center-to-point ray; for the axis-aligned ellipsoids
    used here this is not the Euclidean nearest-point projection, but it always
    returns a feasible point and leaves feasible points unchanged). If radius
    is None, no retraction is done.
    """
    if radius is None or L_inv is None:
        return x
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive when a trust region is used")
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    L_inv = np.asarray(L_inv, dtype=float)
    if center.shape != x.shape or L_inv.shape != (x.size, x.size):
        raise ValueError("center and L_inv must match the parameter dimension")
    d = x - center
    m = L_inv @ d
    dist = np.linalg.norm(m)
    if dist <= radius or dist == 0:
        return x
    return center + (radius / dist) * d


def coordinate_search(objective, x0, budget, step0=0.3, center=None,
                      L_inv=None, radius=None, rng=None, target=None,
                      step_scale=None, candidates=None, recenter_on_best=False):
    """Derivative-free coordinate/pattern search with query budget.

    Returns (best_params, best_value, evaluations_used). Respects an optional
    Mahalanobis trust region (center, L_inv, radius) by projecting every
    candidate back into the ellipsoid before evaluation. `step_scale` (per-
    coordinate multipliers derived from the predicted covariance) preconditions
    the search: confident directions take small steps, uncertain directions take
    larger ones, so the learned geometry -- not just a point -- drives the
    trajectory. If `target` is given, search stops as soon as the objective
    reaches it; the query count is then the operational cost to reach that
    quality.

    `candidates` is an optional list of extra seed points. Exact duplicate seed
    vectors are removed before the seed set is evaluated, with one counted query
    for each remaining seed. The best becomes the starting point and best-so-far,
    so the search is never worse than the best evaluated prior it is handed (for
    example, the concentration heuristic). With `recenter_on_best`, the
    Mahalanobis trust region is re-centered on the winning seed and the seeds are
    evaluated unprojected, so when the heuristic seed wins the subsequent search
    explores its basin while retaining the learned shape and preconditioning.
    """
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    if not hasattr(objective, "count"):
        raise TypeError("objective must expose a mutable count")
    if not np.isfinite(step0) or step0 <= 0:
        raise ValueError("step0 must be positive")
    if target is not None and not np.isfinite(target):
        raise ValueError("target must be finite when supplied")
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.array(x0, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    dim = len(x)
    if step_scale is None:
        step_scale = np.ones(dim)
    else:
        step_scale = np.asarray(step_scale, dtype=float)
        if (step_scale.shape != x.shape or not np.all(np.isfinite(step_scale))
                or np.any(step_scale < 0) or not np.any(step_scale > 0)):
            raise ValueError(
                "step_scale must match x0 and contain finite nonnegative values")
    if center is not None:
        center = np.asarray(center, dtype=float)
        if center.shape != x.shape or not np.all(np.isfinite(center)):
            raise ValueError("center must be finite and have the same shape as x0")
    if L_inv is not None:
        L_inv = np.asarray(L_inv, dtype=float)
        if L_inv.shape != (dim, dim) or not np.all(np.isfinite(L_inv)):
            raise ValueError("L_inv must be a finite square matrix matching x0")
    # Assemble the seed set (x0 first, then any extra candidates) and evaluate
    # each exactly once, tracking the best. Seeds are projected into the trust
    # region unless we intend to re-center it on the winner.
    seeds = []
    for seed in [x] + [np.asarray(c, dtype=float) for c in (candidates or [])]:
        if seed.shape != x.shape or not np.all(np.isfinite(seed)):
            raise ValueError("every candidate must be finite and match x0 shape")
        if not any(np.array_equal(seed, prior) for prior in seeds):
            seeds.append(seed.copy())
    project_seeds = center is not None and not recenter_on_best
    fbest = -np.inf
    xbest = seeds[0]
    for s in seeds:
        if objective.count >= budget:
            break
        s = _project_trust_region(s, center, L_inv, radius) if project_seeds else s
        fs = objective(s)
        if fs > fbest:
            fbest = fs
            xbest = s
        if target is not None and fbest >= target:
            return xbest, fbest, objective.count
    x = xbest
    if recenter_on_best and center is not None:
        center = np.array(x, dtype=float)
    step = step0
    while objective.count < budget and step > 1e-3:
        improved = False
        for d in range(dim):
            for sgn in (+1.0, -1.0):
                if objective.count >= budget:
                    break
                cand = x.copy()
                cand[d] += sgn * step * step_scale[d]
                if center is not None:
                    cand = _project_trust_region(cand, center, L_inv, radius)
                fval = objective(cand)
                if fval > fbest:
                    fbest = fval
                    x = cand
                    improved = True
                if target is not None and fbest >= target:
                    return x, fbest, objective.count
        if not improved:
            step *= 0.5
    return x, fbest, objective.count


def run_policy(C, n, maxcut, init_params, budget, center=None, L_inv=None,
               radius=None, restarts=1, rng=None, sample_seed=0, target=None,
               step_scale=None, step0=0.3, candidates=None,
               recenter_on_best=False, shots=None):
    """Run coordinate search (optionally with restarts) and report metrics.

    Returns both ``evaluations`` (the legacy raw-use key) and
    ``evaluations_used``, plus ``reached_target``, sampled ratio, the exact
    expectation ``value`` of the returned angles, and the best parameters.
    A benchmark that gives one method an early cap must score a miss at the
    shared cap outside this function; raw use alone is not a fair cost-to-target
    score. ``restarts`` random initializations are used for the random-restart
    baseline; each restart shares the total budget. If `target` (an objective
    value) is given, search stops when reached and the evaluation count is the
    cost-to-target.

    With finite ``shots`` the optimizer sees only noisy estimates (and the
    stopping rule fires on the noisy estimate, as it would operationally), but
    the reported `value` is always the exact expectation of the returned
    angles, recomputed once outside the query count, so quality is measured
    without estimator noise.
    """
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    if int(restarts) != restarts or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    budget, restarts = int(budget), int(restarts)
    if rng is None:
        rng = np.random.default_rng(0)
    counter = QueryCounter(C, n, maxcut, shots=shots,
                           noise_rng=np.random.default_rng(sample_seed + 1))
    best_params = np.array(init_params, dtype=float)
    best_val = -np.inf
    per = max(1, budget // restarts)
    for r in range(restarts):
        if r == 0:
            x0 = np.array(init_params, dtype=float)
        else:
            x0 = rng.uniform(0, np.pi, size=len(init_params))
            x0[1::2] = rng.uniform(0, np.pi / 2, size=len(init_params) // 2)
        sub_budget = counter.count + per if restarts > 1 else budget
        # extra seed candidates are only used on the first start
        cand = candidates if r == 0 else None
        xr, fr, _ = coordinate_search(counter, x0, min(sub_budget, budget),
                                      step0=step0, center=center, L_inv=L_inv,
                                      radius=radius, rng=rng, target=target,
                                      step_scale=step_scale, candidates=cand,
                                      recenter_on_best=recenter_on_best)
        if fr > best_val:
            best_val = fr
            best_params = xr
        if counter.count >= budget:
            break
        if target is not None and best_val >= target:
            break
    ratio = best_bitstring_ratio(C, best_params, n, maxcut,
                                 rng=np.random.default_rng(sample_seed))
    # exact quality of the returned angles (uncounted diagnostic; equals
    # best_val in the noiseless case, replaces the noisy estimate otherwise)
    value_exact = qaoa_expectation(C, np.asarray(best_params, dtype=float), n)
    return {
        "evaluations": counter.count,
        "evaluations_used": counter.count,
        "reached_target": (None if target is None
                           else bool(best_val >= target)),
        "ratio": ratio,
        "value": value_exact,
        "params": best_params,
    }


def optimize_callback(evaluator, init_params, budget, center=None, L_inv=None,
                      radius=None, target=None, step_scale=None, step0=0.3,
                      candidates=None, recenter_on_best=False,
                      exact_evaluator=None):
    """Optimize an arbitrary scalar evaluator with counted pattern search.

    ``evaluator`` is the operational, possibly noisy objective and is called at
    most ``budget`` times. If ``exact_evaluator`` is supplied, it is called once
    on the returned parameters to provide an uncounted diagnostic in ``value``;
    the best operational value remains available as ``observed_value``.
    """
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    x0 = np.asarray(init_params, dtype=float)
    if x0.ndim != 1 or x0.size == 0 or not np.all(np.isfinite(x0)):
        raise ValueError("init_params must be a finite one-dimensional vector")
    counter = CallableQueryCounter(evaluator)
    params, observed, _ = coordinate_search(
        counter, x0, int(budget), step0=step0, center=center, L_inv=L_inv,
        radius=radius, target=target, step_scale=step_scale,
        candidates=candidates, recenter_on_best=recenter_on_best,
    )
    value = observed
    if exact_evaluator is not None:
        if not callable(exact_evaluator):
            raise TypeError("exact_evaluator must be callable")
        value = float(exact_evaluator(np.asarray(params, dtype=float)))
        if not np.isfinite(value):
            raise ValueError("exact_evaluator returned a non-finite value")
    return {
        "evaluations": counter.count,
        "evaluations_used": counter.count,
        "reached_target": (None if target is None
                           else bool(observed >= target)),
        "observed_value": float(observed),
        "value": float(value),
        "params": np.asarray(params, dtype=float),
    }
