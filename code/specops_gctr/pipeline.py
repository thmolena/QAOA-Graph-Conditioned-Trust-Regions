"""End-to-end REAL pipeline: data -> targets -> train GNN -> evaluate.

This is the driver behind the `gctr-reproduce` console entrypoint. It runs a
genuine QAOA statevector simulation, trains a torch GNN that predicts a
graph-conditioned Gaussian over angles, fits a held-out error calibrator, and
measures query efficiency (with paired Wilcoxon tests), calibration,
cross-size generalization, leave-one-family-out transfer, the component
ablation, seed stability and shot-noise robustness. Everything is
deterministic given the seed and a fixed environment; nothing is hand-typed.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import networkx as nx
import torch
from scipy.stats import wilcoxon

from .graphs import (generate_dataset, laplacian_spectral_features,
                     graph_feature_dim, FAMILIES)
from .targets import best_angles
from .model import (GraphConditionedGaussian, ErrorCalibrator,
                    build_dense_batch)
from .losses import gaussian_nll, wasserstein2_diag, contrastive_uncertainty
from . import baselines as B
from .calibration import (expected_calibration_error,
                          spearman_uncertainty_error, calibration_curve)
from .qaoa import qaoa_expectation

METHOD_ORDER = ["Random", "Heuristic", "k-NN", "TQA", "GNN point", "GCTR"]


@dataclass
class Config:
    train_n: int = 14
    per_family_train: int = 60
    per_family_val: int = 20
    per_family_test: int = 12
    epochs: int = 120
    hidden_dim: int = 48
    num_layers: int = 3
    p_depth: int = 2
    spectral_k: int = 6
    lr: float = 5e-3
    w_wasserstein: float = 0.3
    w_contrastive: float = 0.2
    budget: int = 400
    target_frac: float = 0.98
    radius_scale: float = 2.0
    random_restarts: int = 10
    n_target_starts: int = 8
    target_budget: int = 60
    budget_cap_min: int = 40
    cross_sizes: tuple = (8, 10, 12, 14, 16)
    stability_seeds: tuple = (20260424, 1, 2, 3)
    shot_levels: tuple = (128, 1024)
    seed: int = 20260424
    use_spectral: bool = True

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def _adj(inst):
    return nx.to_numpy_array(inst.graph, dtype=np.float32)


def _featurize(dataset, k):
    return [laplacian_spectral_features(inst.graph, k) for inst in dataset]


def concentration_angles(tr_ang):
    """The parameter-concentration heuristic's angles: the coordinate-wise
    median of the training set's target angles."""
    return np.median(np.asarray(tr_ang, dtype=float), axis=0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def budget_rule(cfg: Config, u, u_med, u_iqr):
    """Uncertainty -> per-instance search allocation.

    The calibrated predicted uncertainty u is robustly standardized with the
    validation median and interquartile range, z = (u - med)/IQR. The rule is
    monotone and capped:

      K(z) = clip(floor(1 + 4*sigmoid(z)), 1, 5)   total initial seed points
      T(z) = clip(floor(T_base*(0.5 + max(z,0))), T_min, budget)

    with T_base = budget/2. Low-uncertainty instances get a lean seed set and a
    tight evaluation cap; high-uncertainty instances get up to five seeds and
    the full shared cap. The GNN mean and the concentration seed are the first
    two seed points, so n_gaussian_seeds = max(0, K - 2).
    """
    z = (float(u) - float(u_med)) / max(float(u_iqr), 1e-9)
    K = int(np.clip(np.floor(1 + 4 * sigmoid(z)), 1, 5))
    t_base = cfg.budget // 2
    T = int(np.clip(np.floor(t_base * (0.5 + max(z, 0.0))),
                    cfg.budget_cap_min, cfg.budget))
    return dict(z=float(z), K=K, T=T, n_gaussian_seeds=max(0, K - 2))


def prepare(cfg: Config, verbose=True):
    """Generate train/validation/test data with target angles + difficulty.

    The validation split is disjoint from training and is used only to fit the
    error calibrator and the budget-rule standardization statistics; every
    reported metric comes from the test split.
    """
    train = generate_dataset(cfg.train_n, cfg.per_family_train, seed0=100)
    val = generate_dataset(cfg.train_n, cfg.per_family_val, seed0=5000)
    test = generate_dataset(cfg.train_n, cfg.per_family_test, seed0=9000)
    if verbose:
        print(f"[data] train={len(train)} val={len(val)} test={len(test)} "
              f"at n={cfg.train_n}")

    def label(dataset, tag, off):
        angles, errs, vals = [], [], []
        for i, inst in enumerate(dataset):
            a, v, e = best_angles(inst, n_starts=cfg.n_target_starts,
                                  budget_per_start=cfg.target_budget,
                                  seed=cfg.seed + off + i)
            angles.append(a); errs.append(e); vals.append(v)
        if verbose:
            print(f"[targets] {tag}: mean approx-error {np.mean(errs):.3f}")
        return np.array(angles), np.array(errs), np.array(vals)

    tr_ang, tr_err, tr_val = label(train, "train", 0)
    va_ang, va_err, va_val = label(val, "val", 40000)
    te_ang, te_err, te_val = label(test, "test", 80000)
    return dict(train=train, val=val, test=test,
                tr_ang=tr_ang, tr_err=tr_err, tr_val=tr_val,
                va_ang=va_ang, va_err=va_err, va_val=va_val,
                te_ang=te_ang, te_err=te_err, te_val=te_val)


def train_model(cfg: Config, data, use_spectral=None, verbose=True):
    if use_spectral is None:
        use_spectral = cfg.use_spectral
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)  # cross-machine numerical stability
    np.random.seed(cfg.seed)
    k = cfg.spectral_k
    fdim = graph_feature_dim(k)
    train = data["train"]
    feats = _featurize(train, k)
    adjs = [_adj(i) for i in train]
    X, A, M = build_dense_batch(feats, adjs)
    y = torch.tensor(data["tr_ang"], dtype=torch.float32)
    diff = torch.tensor(data["tr_err"], dtype=torch.float32)

    model = GraphConditionedGaussian(fdim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.p_depth, use_spectral=use_spectral)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # target spread for the Wasserstein regularizer: scale with difficulty
    tgt_logvar = torch.log(0.02 + 0.5 * diff).unsqueeze(-1).repeat(1, 2 * cfg.p_depth)
    for ep in range(cfg.epochs):
        model.train(); opt.zero_grad()
        mu, logvar = model(X, A, M)
        loss = gaussian_nll(mu, logvar, y)
        loss = loss + cfg.w_wasserstein * wasserstein2_diag(mu, logvar, y, tgt_logvar)
        loss = loss + cfg.w_contrastive * contrastive_uncertainty(logvar, diff)
        loss.backward(); opt.step()
        if verbose and (ep + 1) % 40 == 0:
            print(f"[train] epoch {ep+1}/{cfg.epochs} loss {loss.item():.3f}")
    # fit the post-hoc heteroscedastic error calibrator on HELD-OUT residuals
    # (the realized objective error of the GNN warm start on the validation
    # split, which the GNN never trained on) so the reported uncertainty
    # tracks the error a fresh instance actually incurs.
    _attach_error_calibrator(model, cfg, data, verbose=verbose)
    return model, feats, adjs


def _realized_obj_error(dataset, mu):
    """Realized objective error 1 - <C>(mu)/maxcut of each warm-start mean."""
    out = []
    for i, inst in enumerate(dataset):
        e = qaoa_expectation(inst.C, mu[i], inst.n)
        out.append(1.0 - e / inst.maxcut if inst.maxcut > 0 else 1.0)
    return np.array(out, dtype=float)


def _attach_error_calibrator(model, cfg, data, verbose=False):
    """Fit a linear calibrator embedding -> log realized objective error on the
    held-out validation split, and attach it (plus a bound
    ``predicted_uncertainty`` method and the budget-rule standardization
    statistics) to ``model``."""
    val = data.get("val") or data["train"]  # fallback for reduced configs
    k = cfg.spectral_k
    feats = _featurize(val, k)
    adjs = [_adj(i) for i in val]
    Xv, Av, Mv = build_dense_batch(feats, adjs)
    model.eval()
    with torch.no_grad():
        pooled = model.embed(Xv, Av, Mv)
        mu_val = model.mu_head(pooled).numpy()
        logvar_val = model.logvar_head(pooled).clamp(-8.0, 2.0).numpy()
    r_val = _realized_obj_error(val, mu_val)
    emb_mean = pooled.mean(dim=0, keepdim=True)
    emb_std = pooled.std(dim=0, keepdim=True) + 1e-6
    pooled_n = (pooled - emb_mean) / emb_std
    target = torch.tensor(np.log(r_val + 1e-4), dtype=torch.float32)

    torch.manual_seed(cfg.seed)
    calib = ErrorCalibrator(pooled.shape[1])
    opt = torch.optim.Adam(calib.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(500):
        opt.zero_grad()
        pred = calib(pooled_n)
        loss = ((pred - target) ** 2).mean()
        loss.backward(); opt.step()

    model.error_calibrator = calib
    model._emb_mean = emb_mean
    model._emb_std = emb_std

    def predicted_uncertainty(dataset, k):
        feats = _featurize(dataset, k)
        adjs = [_adj(i) for i in dataset]
        Xd, Ad, Md = build_dense_batch(feats, adjs)
        model.eval()
        with torch.no_grad():
            p = (model.embed(Xd, Ad, Md) - model._emb_mean) / model._emb_std
            log_err = model.error_calibrator(p).numpy()
        return np.exp(log_err)

    model.predicted_uncertainty = predicted_uncertainty

    # budget-rule robust standardization statistics from the validation split
    with torch.no_grad():
        p_n = (pooled - emb_mean) / emb_std
        u_val = np.exp(calib(p_n).numpy())
    q25, q75 = np.percentile(u_val, [25, 75])
    model._u_med = float(np.median(u_val))
    model._u_iqr = float(max(q75 - q25, 1e-9))
    # tr(Sigma) statistics for the no-calibrator ablation
    trsig_val = np.exp(logvar_val).sum(axis=1)
    q25s, q75s = np.percentile(trsig_val, [25, 75])
    model._trsig_med = float(np.median(trsig_val))
    model._trsig_iqr = float(max(q75s - q25s, 1e-9))
    if verbose:
        print(f"[calib-head] fitted linear error calibrator on {len(val)} "
              "held-out validation residuals")


def predict(model, dataset, k):
    feats = _featurize(dataset, k)
    adjs = [nx.to_numpy_array(i.graph, dtype=np.float32) for i in dataset]
    X, A, M = build_dense_batch(feats, adjs)
    model.eval()
    with torch.no_grad():
        mu, logvar = model(X, A, M)
    return mu.numpy(), np.exp(logvar.numpy()), feats


def gctr_allocation(cfg: Config, model, dataset, use_calibrator=True):
    """Per-instance budget allocation for a dataset: z, K, T per instance."""
    if use_calibrator and hasattr(model, "predicted_uncertainty"):
        u = np.asarray(model.predicted_uncertainty(dataset, cfg.spectral_k))
        med, iqr = model._u_med, model._u_iqr
    else:  # ablation: drive the rule with tr(Sigma) instead
        _, var, _ = predict(model, dataset, cfg.spectral_k)
        u = var.sum(axis=1)
        med = getattr(model, "_trsig_med", float(np.median(u)))
        iqr = getattr(model, "_trsig_iqr", 1e-9)
    return [budget_rule(cfg, ui, med, iqr) for ui in u]


def run_gctr(cfg: Config, inst, mu_i, var_i, alloc, conc, ss, target,
             shots=None, **overrides):
    """One GCTR run with the standard wiring (seeds, budget, trust region)."""
    kw = dict(radius_scale=cfg.radius_scale, sample_seed=ss, target=target,
              heuristic_angles=conc, n_gaussian_seeds=alloc["n_gaussian_seeds"],
              budget_cap=alloc["T"], shots=shots)
    kw.update(overrides)
    return B.gctr_policy(inst, cfg.budget, mu_i, var_i, **kw)


def evaluate_queries(cfg: Config, data, model, verbose=True):
    """Run every baseline + GCTR on the test set; report per-method stats.

    Reports evaluations to reach the shared quality target, wall-clock runtime
    (measured, milliseconds; machine-dependent by nature), the expectation
    ratio <C>/C_max of the returned angles, and the sampled best-bitstring
    ratio (512 Born-rule samples). Also reports two-sided paired Wilcoxon
    signed-rank tests of GCTR against every baseline on the per-instance
    evaluation counts (key "_wilcoxon"), and the per-instance GCTR budget
    allocations (key "_allocation").
    """
    test = data["test"]
    train = data["train"]
    mu, var, te_feats = predict(model, test, cfg.spectral_k)
    tr_feats = _featurize(train, cfg.spectral_k)
    tr_ang = data["tr_ang"]
    conc = concentration_angles(tr_ang)
    allocs = gctr_allocation(cfg, model, test)

    rows = {m: {"evals": [], "ratio": [], "expratio": [], "ms": []}
            for m in METHOD_ORDER}
    for i, inst in enumerate(test):
        target = cfg.target_frac * data["te_val"][i]
        ss = cfg.seed + i

        def timed(fn, *a, **kw):
            t0 = time.perf_counter()
            res = fn(*a, **kw)
            res["ms"] = (time.perf_counter() - t0) * 1e3
            return res

        r = timed(B.random_restart, inst, cfg.budget,
                  restarts=cfg.random_restarts,
                  rng=np.random.default_rng(ss), sample_seed=ss, target=target)
        h = timed(B.heuristic_init, inst, cfg.budget, angles=conc,
                  sample_seed=ss, target=target)
        kn = timed(B.knn_init, inst, cfg.budget, tr_feats, tr_ang, te_feats[i],
                   sample_seed=ss, target=target)
        tq = timed(B.tqa_init, inst, cfg.budget, sample_seed=ss, target=target)
        gp = timed(B.gnn_point, inst, cfg.budget, mu[i], sample_seed=ss,
                   target=target)
        gc = timed(run_gctr, cfg, inst, mu[i], var[i], allocs[i], conc, ss,
                   target)
        for name, res in zip(METHOD_ORDER, [r, h, kn, tq, gp, gc]):
            rows[name]["evals"].append(res["evaluations"])
            rows[name]["ratio"].append(res["ratio"])
            rows[name]["expratio"].append(
                res["value"] / inst.maxcut if inst.maxcut > 0 else 0.0)
            rows[name]["ms"].append(res["ms"])
    summary = {}
    for name, d in rows.items():
        ev = np.array(d["evals"]); ra = np.array(d["ratio"])
        er = np.array(d["expratio"]); ms = np.array(d["ms"])
        summary[name] = dict(evals_mean=float(ev.mean()), evals_sd=float(ev.std()),
                             ratio_mean=float(ra.mean()), ratio_sd=float(ra.std()),
                             expratio_mean=float(er.mean()),
                             expratio_sd=float(er.std()),
                             ms_mean=float(ms.mean()), ms_sd=float(ms.std()),
                             evals=[int(x) for x in d["evals"]])
        if verbose:
            print(f"[queries] {name:10s} evals={ev.mean():6.1f}+-{ev.std():4.1f} "
                  f"expratio={er.mean():.3f} sampled={ra.mean():.3f} "
                  f"ms={ms.mean():.0f}")

    # paired two-sided Wilcoxon signed-rank tests: GCTR vs each baseline
    gctr_ev = np.array(rows["GCTR"]["evals"], dtype=float)
    stats = {}
    for name in METHOD_ORDER:
        if name == "GCTR":
            continue
        base_ev = np.array(rows[name]["evals"], dtype=float)
        try:
            stat, p = wilcoxon(gctr_ev, base_ev)
            stats[name] = dict(w=float(stat), p=float(p), n=len(gctr_ev))
        except ValueError:  # all differences are zero
            stats[name] = dict(w=0.0, p=1.0, n=len(gctr_ev))
        if verbose:
            print(f"[wilcoxon] GCTR vs {name:10s} p={stats[name]['p']:.4g}")
    summary["_wilcoxon"] = stats
    summary["_allocation"] = dict(
        z=[a["z"] for a in allocs], K=[a["K"] for a in allocs],
        T=[a["T"] for a in allocs],
        n_gaussian_seeds=[a["n_gaussian_seeds"] for a in allocs])
    return summary, mu, var


def evaluate_calibration(cfg: Config, data, model, verbose=True,
                         use_calibrator=True):
    """ECE + Spearman on the test set (real bin counts).

    ECE is the coverage expected calibration error of the Gaussian heads (mean
    |empirical - nominal| coverage over ten levels). The headline Spearman rho
    correlates the *calibrated* predicted uncertainty (the error-calibrator
    head, fit on held-out validation residuals) with the realized objective
    error of the GNN warm start on the test set, and is reported WITH its
    p-value and sample size. For transparency we also report the legacy
    quantity: the rank correlation of tr(Sigma) with the offline instance
    difficulty.
    """
    test = data["test"]
    mu, var, feats = predict(model, test, cfg.spectral_k)
    te_ang = data["te_ang"]
    sigma = np.sqrt(var)
    z = ((te_ang - mu) / sigma).reshape(-1)
    ece = expected_calibration_error(z, n_bins=10)

    # realized objective error of the warm-start mean (ground-truth error)
    realized_err = _realized_obj_error(test, mu)
    fn = getattr(model, "predicted_uncertainty", None)
    if use_calibrator and fn is not None:
        pred_unc = np.asarray(fn(test, cfg.spectral_k))
    else:  # ablation / original method: fall back to tr(Sigma)
        pred_unc = var.sum(axis=1)
    rho, rho_p, rho_n = spearman_uncertainty_error(pred_unc, realized_err)

    # legacy metric (tr(Sigma) vs offline difficulty), kept for continuity
    rho_legacy, rho_legacy_p, _ = spearman_uncertainty_error(
        var.sum(axis=1), data["te_err"])

    pred, obs, counts, N = calibration_curve(z, n_bins=10)
    if verbose:
        print(f"[calib] ECE={ece:.3f} Spearman rho={rho:.3f} (p={rho_p:.3g}, "
              f"n={rho_n}) [legacy tr(Sigma) vs difficulty={rho_legacy:.3f}] "
              f"(N residuals={len(z)})")
    return dict(ece=float(ece), spearman=float(rho),
                spearman_p=float(rho_p), spearman_n=int(rho_n),
                spearman_legacy=float(rho_legacy),
                spearman_legacy_p=float(rho_legacy_p),
                trsigma_spread=float(np.ptp(var.sum(axis=1))),
                curve=dict(predicted=pred.tolist(), observed=obs.tolist(),
                           counts=counts.tolist(), N=int(N)))


def evaluate_generalization(cfg: Config, data, model, verbose=True):
    """Cross-size transfer: evaluations to target at each size for random
    restarts, the concentration heuristic, the GNN point baseline and GCTR
    (model and heuristic angles fixed from the training size)."""
    conc = concentration_angles(data["tr_ang"])
    out = {}
    for n in cfg.cross_sizes:
        ds = generate_dataset(n, cfg.per_family_test, seed0=50000 + n)
        # need target values per instance
        vals = []
        for i, inst in enumerate(ds):
            _, v, _ = best_angles(inst, n_starts=cfg.n_target_starts,
                                  budget_per_start=cfg.target_budget,
                                  seed=cfg.seed + n + i)
            vals.append(v)
        mu, var, feats = predict(model, ds, cfg.spectral_k)
        allocs = gctr_allocation(cfg, model, ds)
        r_ev, h_ev, p_ev, g_ev = [], [], [], []
        for i, inst in enumerate(ds):
            target = cfg.target_frac * vals[i]
            ss = cfg.seed + n + i
            r = B.random_restart(inst, cfg.budget, restarts=cfg.random_restarts,
                                 rng=np.random.default_rng(ss), sample_seed=ss,
                                 target=target)
            h = B.heuristic_init(inst, cfg.budget, angles=conc, sample_seed=ss,
                                 target=target)
            gp = B.gnn_point(inst, cfg.budget, mu[i], sample_seed=ss,
                             target=target)
            gc = run_gctr(cfg, inst, mu[i], var[i], allocs[i], conc, ss, target)
            r_ev.append(r["evaluations"]); h_ev.append(h["evaluations"])
            p_ev.append(gp["evaluations"]); g_ev.append(gc["evaluations"])
        speedup = float(np.mean(r_ev) / max(1e-9, np.mean(g_ev)))
        out[n] = dict(random_evals=float(np.mean(r_ev)),
                      heuristic_evals=float(np.mean(h_ev)),
                      point_evals=float(np.mean(p_ev)),
                      gctr_evals=float(np.mean(g_ev)), speedup=speedup)
        if verbose:
            print(f"[general] n={n:2d} speedup={speedup:.2f}x "
                  f"(rand {np.mean(r_ev):.1f} / heur {np.mean(h_ev):.1f} / "
                  f"point {np.mean(p_ev):.1f} / gctr {np.mean(g_ev):.1f})")
    return out


def evaluate_lofo(cfg: Config, data, verbose=True):
    """Leave-one-family-out transfer: retrain without one family, evaluate on it.

    For each graph family, the model (and its calibrator/validation statistics)
    is retrained on the remaining three families' training and validation
    instances, the concentration angles are recomputed from the reduced
    training set, and the full GCTR policy (plus the point baseline and the
    concentration heuristic, for reference) is evaluated on the held-out
    family's test instances under the same cost-to-target protocol. Two policy
    ablations run in the same setting so the trust region's and the adaptive
    budget's contributions under family shift are measured, not asserted:
    GCTR without the Mahalanobis constraint, and GCTR with the allocation
    frozen at its default (no uncertainty-dependent seeds or cap).
    """
    per_fam = {}
    all_gc, all_gp, all_h, all_nt, all_fb = [], [], [], [], []
    fixed_alloc = dict(z=0.0, K=2, T=None, n_gaussian_seeds=0)
    for fam in FAMILIES:
        tr_idx = [i for i, inst in enumerate(data["train"]) if inst.family != fam]
        va_idx = [i for i, inst in enumerate(data["val"]) if inst.family != fam]
        te_idx = [i for i, inst in enumerate(data["test"]) if inst.family == fam]
        sub = dict(train=[data["train"][i] for i in tr_idx],
                   tr_ang=data["tr_ang"][tr_idx],
                   tr_err=data["tr_err"][tr_idx],
                   tr_val=data["tr_val"][tr_idx],
                   val=[data["val"][i] for i in va_idx],
                   va_ang=data["va_ang"][va_idx],
                   va_err=data["va_err"][va_idx],
                   va_val=data["va_val"][va_idx])
        model, _, _ = train_model(cfg, sub, verbose=False)
        conc = concentration_angles(sub["tr_ang"])
        te_insts = [data["test"][i] for i in te_idx]
        mu, var, _ = predict(model, te_insts, cfg.spectral_k)
        allocs = gctr_allocation(cfg, model, te_insts)
        gc_ev, gp_ev, h_ev, nt_ev, fb_ev = [], [], [], [], []
        for j, inst in enumerate(te_insts):
            target = cfg.target_frac * data["te_val"][te_idx[j]]
            ss = cfg.seed + 777 + te_idx[j]
            gc = run_gctr(cfg, inst, mu[j], var[j], allocs[j], conc, ss, target)
            gp = B.gnn_point(inst, cfg.budget, mu[j], sample_seed=ss,
                             target=target)
            h = B.heuristic_init(inst, cfg.budget, angles=conc, sample_seed=ss,
                                 target=target)
            nt = run_gctr(cfg, inst, mu[j], var[j], allocs[j], conc, ss,
                          target, use_trust_region=False)
            fb = run_gctr(cfg, inst, mu[j], var[j], fixed_alloc, conc, ss,
                          target, budget_cap=None)
            gc_ev.append(gc["evaluations"]); gp_ev.append(gp["evaluations"])
            h_ev.append(h["evaluations"]); nt_ev.append(nt["evaluations"])
            fb_ev.append(fb["evaluations"])
        per_fam[fam] = dict(gctr_evals=float(np.mean(gc_ev)),
                            point_evals=float(np.mean(gp_ev)),
                            heuristic_evals=float(np.mean(h_ev)),
                            gctr_no_tr_evals=float(np.mean(nt_ev)),
                            gctr_fixed_budget_evals=float(np.mean(fb_ev)),
                            n_test=len(te_insts))
        all_gc += gc_ev; all_gp += gp_ev; all_h += h_ev
        all_nt += nt_ev; all_fb += fb_ev
        if verbose:
            print(f"[lofo] held-out {fam:3s}: gctr={np.mean(gc_ev):6.1f} "
                  f"point={np.mean(gp_ev):6.1f} heur={np.mean(h_ev):6.1f} "
                  f"noTR={np.mean(nt_ev):6.1f} fixedB={np.mean(fb_ev):6.1f} "
                  f"(n={len(te_insts)})")
    pooled = dict(gctr_evals_mean=float(np.mean(all_gc)),
                  gctr_evals_sd=float(np.std(all_gc)),
                  point_evals_mean=float(np.mean(all_gp)),
                  point_evals_sd=float(np.std(all_gp)),
                  heuristic_evals_mean=float(np.mean(all_h)),
                  heuristic_evals_sd=float(np.std(all_h)),
                  gctr_no_tr_evals_mean=float(np.mean(all_nt)),
                  gctr_no_tr_evals_sd=float(np.std(all_nt)),
                  gctr_fixed_budget_evals_mean=float(np.mean(all_fb)),
                  gctr_fixed_budget_evals_sd=float(np.std(all_fb)))
    if verbose:
        print(f"[lofo] pooled: gctr={pooled['gctr_evals_mean']:.1f}"
              f"+-{pooled['gctr_evals_sd']:.1f} "
              f"point={pooled['point_evals_mean']:.1f}"
              f"+-{pooled['point_evals_sd']:.1f} "
              f"heur={pooled['heuristic_evals_mean']:.1f} "
              f"noTR={pooled['gctr_no_tr_evals_mean']:.1f} "
              f"fixedB={pooled['gctr_fixed_budget_evals_mean']:.1f}")
    return dict(per_family=per_fam, pooled=pooled)


def evaluate_ablation(cfg: Config, data, verbose=True):
    """Component ablation.

    Policy-level variants reuse the full trained model and switch off one
    inference-time component at a time (trust region, heuristic seed, adaptive
    budget, error calibrator); training-level variants retrain the model with
    one loss/feature removed. Each row reports evaluations to target on the
    test set plus the calibration metrics appropriate to the variant.
    """
    test = data["test"]
    conc = concentration_angles(data["tr_ang"])

    # --- full model, policy-level variants -------------------------------
    model, _, _ = train_model(cfg, data, verbose=False)
    mu, var, _ = predict(model, test, cfg.spectral_k)
    allocs = gctr_allocation(cfg, model, test)
    allocs_trsig = gctr_allocation(cfg, model, test, use_calibrator=False)
    fixed_alloc = dict(z=0.0, K=2, T=None, n_gaussian_seeds=0)

    policy_variants = ["Full method", "No trust region", "No heuristic seed",
                       "No adaptive budget", "No error calibrator"]

    def run_variant(name):
        evs = []
        for i, inst in enumerate(test):
            target = cfg.target_frac * data["te_val"][i]
            ss = cfg.seed + i
            if name == "No trust region":
                res = run_gctr(cfg, inst, mu[i], var[i], allocs[i], conc, ss,
                               target, use_trust_region=False)
            elif name == "No heuristic seed":
                res = run_gctr(cfg, inst, mu[i], var[i], allocs[i], conc, ss,
                               target, use_heuristic_seed=False)
            elif name == "No adaptive budget":
                res = run_gctr(cfg, inst, mu[i], var[i], fixed_alloc, conc, ss,
                               target, budget_cap=None)
            elif name == "No error calibrator":
                res = run_gctr(cfg, inst, mu[i], var[i], allocs_trsig[i], conc,
                               ss, target)
            else:  # Full method
                res = run_gctr(cfg, inst, mu[i], var[i], allocs[i], conc, ss,
                               target)
            evs.append(res["evaluations"])
        return evs

    results = {}
    for name in policy_variants:
        evs = run_variant(name)
        calib = evaluate_calibration(
            cfg, data, model, verbose=False,
            use_calibrator=(name != "No error calibrator"))
        results[name] = dict(evals_mean=float(np.mean(evs)),
                             evals_sd=float(np.std(evs)),
                             ece=calib["ece"], spearman=calib["spearman"],
                             spearman_p=calib["spearman_p"])
        if verbose:
            print(f"[ablation] {name:22s} evals={np.mean(evs):5.1f} "
                  f"ECE={calib['ece']:.3f} rho={calib['spearman']:.3f}")

    # --- retrained variants (loss/feature removed) ------------------------
    retrain_variants = {
        "No Wasserstein loss": dict(w_wasserstein=0.0),
        "No ranking penalty": dict(w_contrastive=0.0),
        "No spectral encodings": dict(use_spectral=False),
    }
    for name, ov in retrain_variants.items():
        c = Config(**{**cfg.to_dict(), **{k: v for k, v in ov.items()
                                          if k != "use_spectral"}})
        c.cross_sizes = tuple(cfg.cross_sizes)
        c.stability_seeds = tuple(cfg.stability_seeds)
        c.shot_levels = tuple(cfg.shot_levels)
        use_spec = ov.get("use_spectral", cfg.use_spectral)
        m2, _, _ = train_model(c, data, use_spectral=use_spec, verbose=False)
        mu2, var2, _ = predict(m2, test, cfg.spectral_k)
        allocs2 = gctr_allocation(cfg, m2, test)
        evs = []
        for i, inst in enumerate(test):
            target = cfg.target_frac * data["te_val"][i]
            ss = cfg.seed + i
            res = run_gctr(cfg, inst, mu2[i], var2[i], allocs2[i], conc, ss,
                           target)
            evs.append(res["evaluations"])
        calib = evaluate_calibration(cfg, data, m2, verbose=False)
        results[name] = dict(evals_mean=float(np.mean(evs)),
                             evals_sd=float(np.std(evs)),
                             ece=calib["ece"], spearman=calib["spearman"],
                             spearman_p=calib["spearman_p"])
        if verbose:
            print(f"[ablation] {name:22s} evals={np.mean(evs):5.1f} "
                  f"ECE={calib['ece']:.3f} rho={calib['spearman']:.3f}")
    return results


def evaluate_seed_stability(cfg: Config, data, verbose=True):
    """Retrain and re-evaluate the query benchmark across several seeds.

    The benchmark instances and their targets stay fixed (they define the
    task); the seed changes model initialization/training and every stochastic
    element of the search. Reports per-seed method means and the cross-seed
    mean +- sd, substantiating the manuscript's multi-seed claims.
    """
    per_seed = {}
    for s in cfg.stability_seeds:
        c = Config(**cfg.to_dict())
        c.cross_sizes = tuple(cfg.cross_sizes)
        c.stability_seeds = tuple(cfg.stability_seeds)
        c.shot_levels = tuple(cfg.shot_levels)
        c.seed = int(s)
        model, _, _ = train_model(c, data, verbose=False)
        summary, _, _ = evaluate_queries(c, data, model, verbose=False)
        calib = evaluate_calibration(c, data, model, verbose=False)
        per_seed[str(s)] = dict(
            queries={m: {k: v for k, v in summary[m].items() if k != "evals"}
                     for m in METHOD_ORDER},
            wilcoxon=summary["_wilcoxon"],
            ece=calib["ece"], spearman=calib["spearman"],
            spearman_p=calib["spearman_p"])
        if verbose:
            q = per_seed[str(s)]["queries"]
            print(f"[seeds] seed={s}: GCTR {q['GCTR']['evals_mean']:.1f} "
                  f"vs Heuristic {q['Heuristic']['evals_mean']:.1f} evals, "
                  f"rho={calib['spearman']:.3f}")
    agg = {}
    for m in METHOD_ORDER:
        ev = [per_seed[str(s)]["queries"][m]["evals_mean"]
              for s in cfg.stability_seeds]
        er = [per_seed[str(s)]["queries"][m]["expratio_mean"]
              for s in cfg.stability_seeds]
        agg[m] = dict(evals_mean=float(np.mean(ev)), evals_sd=float(np.std(ev)),
                      expratio_mean=float(np.mean(er)),
                      expratio_sd=float(np.std(er)))
    rhos = [per_seed[str(s)]["spearman"] for s in cfg.stability_seeds]
    eces = [per_seed[str(s)]["ece"] for s in cfg.stability_seeds]
    agg["_spearman"] = dict(mean=float(np.mean(rhos)), sd=float(np.std(rhos)))
    agg["_ece"] = dict(mean=float(np.mean(eces)), sd=float(np.std(eces)))
    if verbose:
        print(f"[seeds] aggregate GCTR {agg['GCTR']['evals_mean']:.1f}"
              f"+-{agg['GCTR']['evals_sd']:.1f} vs Heuristic "
              f"{agg['Heuristic']['evals_mean']:.1f}"
              f"+-{agg['Heuristic']['evals_sd']:.1f}; "
              f"rho {agg['_spearman']['mean']:.3f}+-{agg['_spearman']['sd']:.3f}")
    return dict(seeds=[int(s) for s in cfg.stability_seeds],
                per_seed=per_seed, aggregate=agg)


def evaluate_shot_noise(cfg: Config, data, model, verbose=True):
    """Finite-shot robustness: rerun the query benchmark with the optimizer
    seeing only S-shot estimates of the objective (S in cfg.shot_levels), for
    random restarts, the concentration heuristic and GCTR. The stopping rule
    fires on the noisy estimate (as it would operationally); the reported
    quality is the exact expectation of the returned angles.
    """
    test = data["test"]
    conc = concentration_angles(data["tr_ang"])
    mu, var, _ = predict(model, test, cfg.spectral_k)
    allocs = gctr_allocation(cfg, model, test)
    out = {}
    for shots in cfg.shot_levels:
        rows = {m: {"evals": [], "expratio": []}
                for m in ["Random", "Heuristic", "GCTR"]}
        for i, inst in enumerate(test):
            target = cfg.target_frac * data["te_val"][i]
            ss = cfg.seed + i
            r = B.random_restart(inst, cfg.budget, restarts=cfg.random_restarts,
                                 rng=np.random.default_rng(ss), sample_seed=ss,
                                 target=target, shots=shots)
            h = B.heuristic_init(inst, cfg.budget, angles=conc, sample_seed=ss,
                                 target=target, shots=shots)
            gc = run_gctr(cfg, inst, mu[i], var[i], allocs[i], conc, ss,
                          target, shots=shots)
            for name, res in zip(["Random", "Heuristic", "GCTR"], [r, h, gc]):
                rows[name]["evals"].append(res["evaluations"])
                rows[name]["expratio"].append(
                    res["value"] / inst.maxcut if inst.maxcut > 0 else 0.0)
        out[int(shots)] = {
            m: dict(evals_mean=float(np.mean(d["evals"])),
                    evals_sd=float(np.std(d["evals"])),
                    expratio_mean=float(np.mean(d["expratio"])),
                    expratio_sd=float(np.std(d["expratio"])))
            for m, d in rows.items()}
        if verbose:
            o = out[int(shots)]
            print(f"[shots] S={shots}: GCTR {o['GCTR']['evals_mean']:.1f} evals "
                  f"@{o['GCTR']['expratio_mean']:.3f} | Heuristic "
                  f"{o['Heuristic']['evals_mean']:.1f}@"
                  f"{o['Heuristic']['expratio_mean']:.3f} | Random "
                  f"{o['Random']['evals_mean']:.1f}@"
                  f"{o['Random']['expratio_mean']:.3f}")
    return out


def landscape_slice(cfg: Config, data, model, grid: int = 41, verbose=True):
    """Compute a REAL 2D landscape slice for one test instance.

    Fixes (gamma_2, beta_2) at the predicted mean and sweeps (gamma_1, beta_1)
    over a grid, evaluating the exact QAOA expectation ratio <C>/C_max at every
    grid point. Returns the grid, the predicted mean/std for the swept
    coordinates (defining the trust-region ellipse cross-section) and the
    instance identity, so the figure is traceable to a concrete graph.
    """
    test = data["test"]
    # pick the instance whose target approx-error is the median (representative)
    order = np.argsort(data["te_err"])
    inst_idx = int(order[len(order) // 2])
    inst = test[inst_idx]
    mu, var, _ = predict(model, [inst], cfg.spectral_k)
    mu = mu[0]; sd = np.sqrt(var[0])
    gammas = np.linspace(0.0, np.pi, grid)
    betas = np.linspace(0.0, np.pi / 2, grid)
    vals = np.zeros((grid, grid))
    for a, g in enumerate(gammas):
        for b, bb in enumerate(betas):
            params = np.array([g, bb, mu[2], mu[3]])
            vals[b, a] = qaoa_expectation(inst.C, params, inst.n) / inst.maxcut
    if verbose:
        print(f"[landscape] instance family={inst.family} seed={inst.seed} "
              f"n={inst.n} grid={grid}x{grid}")
    return dict(gammas=gammas.tolist(), betas=betas.tolist(),
                values=vals.tolist(),
                mu=[float(mu[0]), float(mu[1])],
                sd=[float(sd[0]), float(sd[1])],
                fixed=[float(mu[2]), float(mu[3])],
                radius_scale=cfg.radius_scale,
                instance=dict(family=inst.family, seed=inst.seed, n=inst.n,
                              maxcut=int(inst.maxcut),
                              edges=[list(map(int, e)) for e in
                                     inst.graph.edges()]))


def pipeline_example(cfg: Config, data, model):
    """Concrete example (graph + predicted Gaussian) for the pipeline figure."""
    inst = data["test"][0]
    mu, var, _ = predict(model, [inst], cfg.spectral_k)
    return dict(edges=[list(map(int, e)) for e in inst.graph.edges()],
                n=inst.n, family=inst.family, seed=inst.seed,
                mu=[float(x) for x in mu[0]],
                sd=[float(x) for x in np.sqrt(var[0])])
