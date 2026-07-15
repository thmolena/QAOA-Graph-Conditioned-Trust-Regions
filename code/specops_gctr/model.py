"""Graph isomorphism network predicting a Gaussian over QAOA angles.

The network maps a graph to a diagonal Gaussian N(mu(G), Sigma(G)) over the 2p
QAOA angles. This is a probabilistic surrogate for the expensive quantum
optimization loop: mu initializes local search and Sigma both preconditions the
step sizes and defines a Mahalanobis trust region. A separate post-hoc
ErrorCalibrator head, fit on held-out (validation) residuals, predicts the
realized warm-start error; that calibrated uncertainty is what allocates the
per-instance seed count and evaluation budget (see pipeline.budget_rule), so
the uncertainty is operational, not decorative.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GINLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.ReLU(),
            nn.Linear(dim_out, dim_out), nn.LayerNorm(dim_out),
        )

    def forward(self, h, adj):
        neigh = torch.bmm(adj, h)
        return self.mlp((1.0 + self.eps) * h + neigh)


class GraphConditionedGaussian(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, num_layers=3, p_depth=2,
                 use_spectral=True):
        super().__init__()
        self.p_depth = p_depth
        self.use_spectral = use_spectral
        self.input_dim = input_dim
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(GINLayer(dim, hidden_dim))
            dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.mu_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * p_depth))
        self.logvar_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * p_depth))

    def embed(self, x, adj, mask):
        """Return the pooled (mean||max) graph embedding [B, 2*hidden_dim].

        Exposed so a post-hoc uncertainty calibrator can be fit on the same
        representation the mean/geometry heads use, without recomputing the
        message passing.
        """
        if not self.use_spectral:
            # ablation: keep only the degree column (feature 0), zero the rest
            x = x.clone()
            x[..., 1:] = 0.0
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, adj))
        m = mask.unsqueeze(-1)
        summ = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        mx = (h.masked_fill(m == 0, -1e9)).max(dim=1).values
        return torch.cat([summ, mx], dim=-1)

    def forward(self, x, adj, mask):
        pooled = self.embed(x, adj, mask)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-8.0, 2.0)
        return mu, logvar


class ErrorCalibrator(nn.Module):
    """Post-hoc heteroscedastic error calibrator (improvement A).

    A small linear map from the frozen graph embedding to the *predicted log
    realized error* of the GNN warm start. It is fit on the residuals of a
    held-out validation split (graphs the GNN never trained on), so the
    reported uncertainty tracks the error a practitioner actually incurs --
    unlike tr(Sigma), which the likelihood-driven losses can flatten across
    instances. Kept deliberately linear because only a few dozen labelled
    graphs are available; an MLP/NLL head overfits.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.lin = nn.Linear(embed_dim, 1)

    def forward(self, pooled):
        return self.lin(pooled).squeeze(-1)


def build_dense_batch(feature_list, adj_list, device="cpu"):
    """Pad a list of variable-size graphs into dense [B,N,F], [B,N,N], [B,N]."""
    B = len(feature_list)
    F = feature_list[0].shape[1]
    N = max(f.shape[0] for f in feature_list)
    X = np.zeros((B, N, F), dtype=np.float32)
    A = np.zeros((B, N, N), dtype=np.float32)
    M = np.zeros((B, N), dtype=np.float32)
    for b, (f, a) in enumerate(zip(feature_list, adj_list)):
        nb = f.shape[0]
        X[b, :nb] = f
        A[b, :nb, :nb] = a
        M[b, :nb] = 1.0
    return (torch.tensor(X, device=device),
            torch.tensor(A, device=device),
            torch.tensor(M, device=device))
