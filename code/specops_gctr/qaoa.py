"""Exact-statevector QAOA for unweighted MaxCut.

A quantum computer is a stochastic device: the Born rule returns one bitstring
per run, so an objective value is a statistical estimate. Here we compute the
*exact* expectation (infinite-shot limit) so that the query-efficiency study is
not confounded by shot noise; the number of objective queries is nonetheless the
resource we count, exactly as it would be on hardware.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def maxcut_cost_diagonal(graph: nx.Graph) -> np.ndarray:
    """Diagonal of the MaxCut cost operator C over the 2^n computational basis.

    C = sum_{(i,j) in E} 0.5 (1 - z_i z_j), z in {+1,-1}. Returns a vector of
    length 2^n giving the cut value of each bitstring.
    """
    n = graph.number_of_nodes()
    dim = 1 << n
    idx = np.arange(dim)
    # bit b of state s (qubit 0 is least-significant); z = +1 if bit 0 else -1
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    z = 1 - 2 * bits  # {+1,-1}
    C = np.zeros(dim)
    for i, j in graph.edges():
        C += 0.5 * (1.0 - z[:, i] * z[:, j])
    return C


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i beta sum_q X_q) to a statevector of 2^n amplitudes."""
    cb = np.cos(beta)
    sb = 1j * np.sin(beta)
    for q in range(n):
        psi = psi.reshape(-1, 2, 1 << q)
        a = psi[:, 0, :].copy()
        c = psi[:, 1, :].copy()
        psi[:, 0, :] = cb * a - sb * c
        psi[:, 1, :] = cb * c - sb * a
        psi = psi.reshape(1 << n)
    return psi


def qaoa_statevector(C: np.ndarray, params: np.ndarray, n: int) -> np.ndarray:
    """Prepare the QAOA state for MaxCut given cost diagonal C and angles.

    params = [gamma_1, beta_1, ..., gamma_p, beta_p]. Returns the amplitude
    vector.
    """
    p = len(params) // 2
    psi = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
    for layer in range(p):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        psi = psi * np.exp(-1j * gamma * C)
        psi = _apply_mixer(psi, beta, n)
    return psi


def qaoa_expectation(C: np.ndarray, params: np.ndarray, n: int) -> float:
    """Exact <C> of the QAOA state (the objective value at these angles)."""
    psi = qaoa_statevector(C, params, n)
    prob = np.abs(psi) ** 2
    return float(np.sum(prob * C))


def qaoa_expectation_sampled(C: np.ndarray, params: np.ndarray, n: int,
                             shots: int,
                             rng: np.random.Generator | None = None) -> float:
    """Finite-shot estimate of <C>: the mean cut value of `shots` bitstrings
    drawn from the Born distribution of the QAOA state.

    This is the estimator a hardware run actually computes; it is unbiased with
    variance Var(C(Z))/shots. Used by the shot-noise robustness study.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    psi = qaoa_statevector(C, params, n)
    prob = np.abs(psi) ** 2
    prob = prob / prob.sum()
    samples = rng.choice(len(prob), size=shots, p=prob)
    return float(C[samples].mean())


def best_bitstring_ratio(C: np.ndarray, params: np.ndarray, n: int,
                         maxcut: float, n_samples: int = 512,
                         rng: np.random.Generator | None = None) -> float:
    """Sampled best-bitstring approximation ratio.

    Draws n_samples bitstrings from the QAOA output distribution (the Born rule)
    and returns (best sampled cut) / (exact maxcut) -- the operational quantity a
    practitioner reads off hardware.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    psi = qaoa_statevector(C, params, n)
    prob = np.abs(psi) ** 2
    prob = prob / prob.sum()
    samples = rng.choice(len(prob), size=n_samples, p=prob)
    best = C[samples].max()
    return float(best / maxcut) if maxcut > 0 else 0.0
