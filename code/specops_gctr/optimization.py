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

    `candidates` (improvement B): an optional list of extra seed points. The
    seed set {x0} U candidates is evaluated first (one honest query each, no
    point evaluated twice); the best becomes the starting point and best-so-far,
    so the search is never worse than the best cheap prior it is handed (e.g. the
    concentration heuristic). With `recenter_on_best`, the Mahalanobis trust
    region is re-centered on the winning seed and the seeds are evaluated
    unprojected -- so when the heuristic seed wins, the subsequent search
    explores its basin (keeping the learned shape/preconditioning) instead of
    being dragged back toward a possibly-wrong GNN mean.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.array(x0, dtype=float)
    dim = len(x)
    if step_scale is None:
        step_scale = np.ones(dim)
    else:
        step_scale = np.asarray(step_scale, dtype=float)
    # Assemble the seed set (x0 first, then any extra candidates) and evaluate
    # each exactly once, tracking the best. Seeds are projected into the trust
    # region unless we intend to re-center it on the winner.
    seeds = [x] + [np.array(c, dtype=float) for c in (candidates or [])]
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

    Returns dict with evaluations, ratio (sampled best-bitstring approx ratio),
    the exact expectation `value` of the returned angles, and the best params
    found. `restarts` random initializations are used for the random-restart
    baseline; each restart shares the total budget. If `target` (an objective
    value) is given, search stops when reached and the evaluation count is the
    cost-to-target.

    With finite ``shots`` the optimizer sees only noisy estimates (and the
    stopping rule fires on the noisy estimate, as it would operationally), but
    the reported `value` is always the exact expectation of the returned
    angles, recomputed once outside the query count, so quality is measured
    without estimator noise.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counter = QueryCounter(C, n, maxcut, shots=shots,
                           noise_rng=np.random.default_rng(sample_seed + 1))
    best_params = np.array(init_params, dtype=float)
    best_val = -np.inf
    per = max(4, budget // restarts)
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
        "ratio": ratio,
        "value": value_exact,
        "params": best_params,
    }
