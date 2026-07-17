"""Calibration diagnostics with explicit bin counts.

Certification of a probabilistic surrogate is estimation plus hypothesis
testing: we report the expected calibration error (ECE) of Gaussian coverage,
the full reliability curve it is computed from, and the Spearman rank
correlation (with its p-value and sample size) between predicted uncertainty
and realized optimization error.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr


def calibration_curve(z_scores, n_bins=10):
    """Reliability curve for a scalar predictive Gaussian.

    z_scores: standardized residuals (target-mu)/sigma aggregated over all
    angle dimensions and instances. For nominal two-sided coverage levels c at
    the centers of ``n_bins`` equal bins of (0,1) we compute the empirical
    fraction of |z| below the Gaussian quantile for c.

    Returns (nominal_levels, observed_coverage, covered_counts, N) where
    covered_counts[k] is the number of residuals covered at nominal level k
    (cumulative in the level, NOT disjoint bin memberships).
    """
    z = np.abs(np.asarray(z_scores, dtype=float))
    levels = (np.arange(1, n_bins + 1) - 0.5) / n_bins  # bin centers in (0,1)
    pred, obs, counts = [], [], []
    N = len(z)
    for c in levels:
        # two-sided coverage c -> half-width quantile
        q = norm.ppf(0.5 + c / 2.0)
        covered = z <= q
        pred.append(c)
        obs.append(float(covered.mean()))
        counts.append(int(covered.sum()))
    return np.array(pred), np.array(obs), np.array(counts), N


def expected_calibration_error(z_scores, n_bins=10):
    """Coverage ECE: mean |empirical - nominal| coverage over nominal levels.

    ECE = (1/K) sum_k |obs(c_k) - c_k| over the K nominal coverage levels of
    the reliability curve. Zero for a perfectly calibrated Gaussian; a model
    whose sigma is systematically too small or too large scores high.
    """
    pred, obs, _, _ = calibration_curve(z_scores, n_bins=n_bins)
    return float(np.mean(np.abs(obs - pred)))


def spearman_uncertainty_error(uncertainty, error):
    """Spearman rank correlation between predicted uncertainty and realized
    error. Returns (rho, p_value, n)."""
    rho, p = spearmanr(uncertainty, error)
    return float(rho), float(p), int(len(np.asarray(uncertainty)))
