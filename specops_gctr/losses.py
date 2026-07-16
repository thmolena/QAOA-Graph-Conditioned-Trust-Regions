"""Training losses for the graph-conditioned Gaussian predictor."""
from __future__ import annotations

import torch


def gaussian_nll(mu, logvar, target):
    """Diagonal Gaussian negative log-likelihood of target angles."""
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mu) ** 2 / var).sum(dim=-1).mean()


def wasserstein2_diag(mu, logvar, target_mu, target_logvar):
    """2-Wasserstein^2 between two diagonal Gaussians (regularizer toward a
    calibrated target spread)."""
    s1 = torch.exp(0.5 * logvar)
    s2 = torch.exp(0.5 * target_logvar)
    return ((mu - target_mu) ** 2 + (s1 - s2) ** 2).sum(dim=-1).mean()


def contrastive_uncertainty(logvar, difficulty):
    """Encourage tr(Sigma) to rank-correlate with instance difficulty.

    A soft pairwise ranking loss: for pairs where difficulty_i > difficulty_j,
    push total predicted variance u_i above u_j.
    """
    u = torch.exp(logvar).sum(dim=-1)
    di = difficulty.unsqueeze(0) - difficulty.unsqueeze(1)
    du = u.unsqueeze(0) - u.unsqueeze(1)
    mask = (di > 0).float()
    # want du>0 where di>0: penalize -du
    loss = torch.relu(0.05 - du) * mask
    denom = mask.sum().clamp(min=1)
    return loss.sum() / denom
