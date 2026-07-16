"""Compute high-quality target QAOA angles per instance (training labels).

The GNN is trained to predict, per graph, a Gaussian whose mean is a good angle
set. We obtain the target angles once, offline, with a modest multi-start
optimizer. These evaluations are *not* counted against the test-time query
budget; they are the (expensive) labels a probabilistic surrogate learns to
amortize.
"""
from __future__ import annotations

import numpy as np

from .optimization import QueryCounter, coordinate_search


def best_angles(inst, n_starts=8, budget_per_start=60, seed=0):
    """Return (best_params, best_value, approx_error) for an instance.

    approx_error = 1 - <C>/maxcut, a difficulty proxy used for the contrastive
    ranking loss and the calibration study.
    """
    rng = np.random.default_rng(seed)
    p = 2
    best_params = None
    best_val = -np.inf
    for s in range(n_starts):
        counter = QueryCounter(inst.C, inst.n, inst.maxcut)
        x0 = np.empty(2 * p)
        x0[0::2] = rng.uniform(0, np.pi, size=p)       # gamma
        x0[1::2] = rng.uniform(0, np.pi / 2, size=p)   # beta
        xr, fr, _ = coordinate_search(counter, x0, budget_per_start, rng=rng)
        if fr > best_val:
            best_val = fr
            best_params = xr
    err = 1.0 - best_val / inst.maxcut if inst.maxcut > 0 else 1.0
    return best_params, best_val, float(err)
