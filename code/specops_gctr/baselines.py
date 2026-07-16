"""Search-policy baselines, all measured in objective evaluations.

- random: random-restart coordinate search (no learned prior).
- heuristic: parameter-concentration heuristic (the coordinate-wise median of
  the training set's target angles, refined locally).
- knn: nearest-neighbour transfer of angles from the training set.
- tqa: Trotterized-quantum-annealing schedule initialization (Sack & Serbyn).
- gnn_point: learned point prediction (GNN mean only, no trust region).
- gctr: full graph-conditioned trust-region policy.

Every method searches until it reaches a shared quality `target` (an objective
value) or exhausts `budget`; the reported cost is the number of objective
evaluations used, exactly the resource that governs hardware runs. All methods
accept `shots` to run under finite-shot estimator noise (see optimization.py).
"""
from __future__ import annotations

import numpy as np

from .optimization import optimize_callback, run_policy, _project_trust_region

# Documented fallback for standalone use of the concentration heuristic. The
# pipeline never uses these values: it passes the coordinate-wise median of the
# training set's target angles (see pipeline.concentration_angles).
FALLBACK_CONCENTRATION_ANGLES = np.array([0.4, 0.6, 0.8, 0.3])


def random_restart(inst, budget, restarts=8, rng=None, sample_seed=0,
                   target=None, shots=None):
    rng = rng or np.random.default_rng(0)
    x0 = rng.uniform(0, np.pi, size=4)
    x0[1::2] = rng.uniform(0, np.pi / 2, size=2)
    return run_policy(inst.C, inst.n, inst.maxcut, x0, budget,
                      restarts=restarts, rng=rng, sample_seed=sample_seed,
                      target=target, shots=shots)


def heuristic_init(inst, budget, angles=None, rng=None, sample_seed=0,
                   target=None, shots=None):
    """Parameter-concentration heuristic: start from fixed angles shared by all
    instances (the pipeline passes the median training target angles)."""
    x0 = np.array(angles if angles is not None
                  else FALLBACK_CONCENTRATION_ANGLES, dtype=float)
    return run_policy(inst.C, inst.n, inst.maxcut, x0, budget,
                      restarts=1, rng=rng or np.random.default_rng(0),
                      sample_seed=sample_seed, target=target, shots=shots)


def tqa_init(inst, budget, rng=None, sample_seed=0, target=None, shots=None):
    """Trotterized-quantum-annealing initialization (Sack & Serbyn 2021): a
    linear ramp evaluated at the layer midpoints, gamma_i = ((i-1/2)/p) dt and
    beta_i = (1-(i-1/2)/p) dt, with dt = 0.75. For p=2 this gives
    [gamma_1, beta_1, gamma_2, beta_2] = [0.1875, 0.5625, 0.5625, 0.1875]."""
    dt = 0.75
    p = 2
    grid = (np.arange(1, p + 1) - 0.5) / p
    x0 = np.empty(2 * p)
    x0[0::2] = grid * dt          # gamma ramps up
    x0[1::2] = (1.0 - grid) * dt  # beta ramps down
    return run_policy(inst.C, inst.n, inst.maxcut, x0, budget,
                      restarts=1, rng=rng or np.random.default_rng(0),
                      sample_seed=sample_seed, target=target, shots=shots)


def knn_init(inst, budget, train_feats, train_angles, feat, rng=None,
             sample_seed=0, target=None, shots=None, k=1):
    """Nearest-neighbour transfer: start from the (mean of the) target angles
    of the k training graphs closest in mean spectral-feature space (k=1 by
    default, i.e. plain nearest-neighbour transfer)."""
    q = feat.mean(axis=0)
    dists = np.array([np.linalg.norm(q - tf.mean(axis=0))
                      for tf in train_feats])
    idx = np.argsort(dists)[:k]
    x0 = np.mean(np.asarray(train_angles, dtype=float)[idx], axis=0)
    return run_policy(inst.C, inst.n, inst.maxcut, x0, budget,
                      restarts=1, rng=rng or np.random.default_rng(0),
                      sample_seed=sample_seed, target=target, shots=shots)


def gnn_point(inst, budget, mu, rng=None, sample_seed=0, target=None,
              shots=None):
    return run_policy(inst.C, inst.n, inst.maxcut, mu, budget,
                      restarts=1, rng=rng or np.random.default_rng(0),
                      sample_seed=sample_seed, target=target, shots=shots)


def _gctr_geometry(mu, sigma_diag, radius_scale, heuristic_angles,
                   use_heuristic_seed, n_gaussian_seeds, sample_seed,
                   use_trust_region):
    """Validate a Gaussian policy and construct its search geometry/seeds."""
    mu = np.asarray(mu, dtype=float)
    sigma_diag = np.asarray(sigma_diag, dtype=float)
    if mu.ndim != 1 or mu.size == 0 or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite one-dimensional vector")
    if sigma_diag.shape != mu.shape or not np.all(np.isfinite(sigma_diag)):
        raise ValueError("sigma_diag must be finite and have the same shape as mu")
    if np.any(sigma_diag <= 0):
        raise ValueError("sigma_diag entries must be positive")
    if radius_scale <= 0:
        raise ValueError("radius_scale must be positive")
    if int(n_gaussian_seeds) != n_gaussian_seeds or n_gaussian_seeds < 0:
        raise ValueError("n_gaussian_seeds must be a nonnegative integer")
    if heuristic_angles is not None:
        heuristic_angles = np.asarray(heuristic_angles, dtype=float)
        if (heuristic_angles.shape != mu.shape or
                not np.all(np.isfinite(heuristic_angles))):
            raise ValueError(
                "heuristic_angles must be finite and have the same shape as mu")

    sigma_diag = np.maximum(sigma_diag, 1e-6)
    std = np.sqrt(sigma_diag)
    L_inv = np.diag(1.0 / std)
    step_scale = std / std.mean()
    radius = radius_scale if use_trust_region else None
    candidates = []
    if use_heuristic_seed and heuristic_angles is not None:
        candidates.append(heuristic_angles.copy())
    if n_gaussian_seeds > 0:
        seed_rng = np.random.default_rng(sample_seed + 7)
        for _ in range(int(n_gaussian_seeds)):
            draw = mu + std * seed_rng.standard_normal(mu.shape)
            draw = _project_trust_region(draw, mu, L_inv, radius)
            candidates.append(draw)
    return mu, L_inv, radius, step_scale, candidates


def gctr_policy(inst, budget, mu, sigma_diag, radius_scale=2.0, rng=None,
                sample_seed=0, target=None, heuristic_angles=None,
                use_heuristic_seed=True, n_gaussian_seeds=0, budget_cap=None,
                use_trust_region=True, shots=None, step0=0.3):
    """Full policy: mean-init + covariance-preconditioned Mahalanobis trust
    region plus score-allocated seeds and budget.

    The predicted covariance does two jobs at once. (i) It preconditions the
    coordinate steps: per-coordinate step sizes are proportional to the
    predicted standard deviation, so confident directions take small,
    exploitative steps and uncertain directions take larger, exploratory ones
    -- the learned geometry, not a fixed step, drives the trajectory. (ii) It
    defines the Mahalanobis ball of radius `radius_scale` (in standard
    deviations) that the search may not leave, so a confident, correct
    prediction contracts the feasible region to a productive basin.

    Seeding ("never worse than the cheap prior"): the search is seeded from the
    GNN mean, optionally from the concentration-heuristic angles
    (`heuristic_angles`; one honest counted query), and from `n_gaussian_seeds`
    draws of the predicted Gaussian retracted into the trust region (one
    counted query each). Best-so-far starts at the best seed and the trust
    region re-centers on it. The returned value is therefore no worse than
    every seed evaluated within the effective budget; this is a quality
    guarantee, not a query-cost guarantee.

    Budget allocation: `n_gaussian_seeds` and `budget_cap` are set per instance
    from the validation-fit difficulty score (see pipeline.budget_rule) --
    confident instances get a lean seed set and a tight evaluation cap,
    uncertain instances get more exploration. `budget_cap` never exceeds
    `budget`, the cap shared by every baseline.

    Ablation switches: `use_heuristic_seed=False` drops the concentration seed;
    `use_trust_region=False` removes the Mahalanobis constraint (keeping seeds,
    preconditioning and budget) to isolate the trust region's contribution.
    """
    mu, L_inv, radius, step_scale, candidates = _gctr_geometry(
        mu, sigma_diag, radius_scale, heuristic_angles, use_heuristic_seed,
        n_gaussian_seeds, sample_seed, use_trust_region)
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    if budget_cap is not None and (int(budget_cap) != budget_cap or budget_cap < 1):
        raise ValueError("budget_cap must be a positive integer when supplied")

    eff_budget = min(budget, budget_cap) if budget_cap else budget
    return run_policy(inst.C, inst.n, inst.maxcut, mu, eff_budget,
                      center=mu if use_trust_region else None,
                      L_inv=L_inv if use_trust_region else None,
                      radius=radius, restarts=1,
                      rng=rng or np.random.default_rng(0),
                      sample_seed=sample_seed, target=target,
                      step_scale=step_scale, step0=step0,
                      candidates=candidates or None,
                      recenter_on_best=bool(candidates), shots=shots)


def gctr_callback_policy(evaluator, budget, mu, sigma_diag, radius_scale=2.0,
                         sample_seed=0, target=None, heuristic_angles=None,
                         use_heuristic_seed=True, n_gaussian_seeds=0,
                         budget_cap=None, use_trust_region=True,
                         exact_evaluator=None, step0=0.3):
    """Run GCTR against an arbitrary scalar objective callback.

    Each call to ``evaluator(parameters)`` is one counted objective query. The
    package makes no assumption about how that scalar is obtained; users remain
    responsible for shot allocation, mitigation, authentication, queueing, and
    backend-specific uncertainty.
    """
    mu, L_inv, radius, step_scale, candidates = _gctr_geometry(
        mu, sigma_diag, radius_scale, heuristic_angles, use_heuristic_seed,
        n_gaussian_seeds, sample_seed, use_trust_region)
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    if budget_cap is not None and (int(budget_cap) != budget_cap or budget_cap < 1):
        raise ValueError("budget_cap must be a positive integer when supplied")
    eff_budget = min(int(budget), int(budget_cap)) if budget_cap else int(budget)
    return optimize_callback(
        evaluator, mu, eff_budget,
        center=mu if use_trust_region else None,
        L_inv=L_inv if use_trust_region else None,
        radius=radius,
        target=target,
        step_scale=step_scale,
        step0=step0,
        candidates=candidates or None,
        recenter_on_best=bool(candidates),
        exact_evaluator=exact_evaluator,
    )
