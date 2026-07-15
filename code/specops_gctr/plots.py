"""Figure builders. Each function reads only the committed source-data files
(CSV + meta.json) written by the real pipeline, and returns a matplotlib
Figure. No values are altered, imputed or fitted; every figure is a
deterministic function of the run outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = {
    "Random": "#9e9e9e",
    "Heuristic": "#8c6bb1",
    "k-NN": "#41ab5d",
    "TQA": "#4292c6",
    "GNN point": "#fe9929",
    "GCTR": "#cb181d",
}
HIGHLIGHT = "GCTR"


def _meta(src: Path) -> dict:
    with open(src / "meta.json") as f:
        return json.load(f)


def _style():
    plt.rcParams.update({
        "font.size": 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "savefig.bbox": "tight",
    })


def fig_query_efficiency(src: Path) -> plt.Figure:
    """Bar chart with 1-sd error bars: mean evaluations to target per method."""
    df = pd.read_csv(src / "Figure2_QueryEfficiency.csv")
    meta = _meta(src)
    n = meta["config"]["train_n"]
    n_test = meta["n_test"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = range(len(df))
    colors = [PALETTE.get(m, "#777777") for m in df["method"]]
    ax.bar(x, df["evaluations"], yerr=df["evaluations_sd"], color=colors,
           capsize=4, edgecolor="black", linewidth=0.6)
    for xi, val in zip(x, df["evaluations"]):
        ax.text(xi, val + 6, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    ax.set_ylabel("Objective (circuit) evaluations to target")
    ax.set_title(rf"Query cost to the $98\%$ target at $n={n}$, $p=2$"
                 rf" ({n_test} test instances)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_calibration(src: Path) -> plt.Figure:
    """Reliability diagram: nominal coverage vs empirical coverage."""
    df = pd.read_csv(src / "Figure3_CalibrationAndUncertainty.csv")
    meta = _meta(src)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="Perfect calibration")
    ax.plot(df["nominal_coverage"], df["observed_coverage"],
            marker="o", color=PALETTE[HIGHLIGHT], label="Observed")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Calibration (ECE = {meta['ece']:.3f}, "
                 f"N = {meta['n_residuals']} residuals)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_generalization(src: Path) -> plt.Figure:
    """Cross-size transfer: evaluations to target vs graph size, per method."""
    df = pd.read_csv(src / "Figure4_Generalization.csv")
    meta = _meta(src)
    n_train = meta["config"]["train_n"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    series = [("random_evals", "Random"), ("heuristic_evals", "Heuristic"),
              ("point_evals", "GNN point"), ("gctr_evals", "GCTR")]
    for col, name in series:
        ax.plot(df["n"], df[col], marker="o",
                linewidth=2.4 if name == HIGHLIGHT else 1.6,
                color=PALETTE[name], label=name)
    ax.set_xlabel("Graph size $n$ (vertices)")
    ax.set_ylabel("Objective evaluations to target")
    ax.set_yscale("log")
    ax.set_title(rf"Cross-size transfer (trained at $n={n_train}$)")
    ax.set_xticks(df["n"])
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_budget_policy(src: Path) -> plt.Figure:
    """Uncertainty-to-budget rule with the realized test-set allocations."""
    df = pd.read_csv(src / "Figure7_BudgetPolicy.csv")
    meta = _meta(src)
    budget = meta["config"]["budget"]
    cap_min = meta["config"].get("budget_cap_min", 40)
    zg = np.linspace(-3, 3, 301)
    K_rule = np.clip(np.floor(1 + 4 / (1 + np.exp(-zg))), 1, 5)
    T_rule = np.clip(np.floor((budget // 2) * (0.5 + np.maximum(zg, 0))),
                     cap_min, budget)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    ax.step(zg, K_rule, where="post", color=PALETTE[HIGHLIGHT], lw=2)
    ax.plot(df["z"], df["K_seeds"], "o", color="#444", ms=5, alpha=0.7,
            label="test instances")
    ax.set_xlabel("standardized uncertainty $z$")
    ax.set_ylabel("initial seed points $K$")
    ax.set_title("Seed allocation $K(z)$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(zg, T_rule, color=PALETTE[HIGHLIGHT], lw=2)
    ax.plot(df["z"], df["T_budget_cap"], "o", color="#444", ms=5, alpha=0.7,
            label="test instances")
    ax.set_xlabel("standardized uncertainty $z$")
    ax.set_ylabel("evaluation cap $T$")
    ax.set_title("Budget cap $T(z)$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Calibrated uncertainty sets the per-instance search budget",
                 fontsize=11)
    fig.tight_layout()
    return fig


def fig_shot_noise(src: Path) -> plt.Figure:
    """Finite-shot robustness: evaluations and quality per shot level."""
    df = pd.read_csv(src / "EDFig2_ShotNoise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    methods = ["Random", "Heuristic", "GCTR"]
    shots = sorted(df["shots"].unique())
    width = 0.25
    x = np.arange(len(shots))
    for j, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("shots").loc[shots]
        axes[0].bar(x + (j - 1) * width, sub["evaluations"], width,
                    yerr=sub["evaluations_sd"], capsize=3, label=m,
                    color=PALETTE[m])
        axes[1].bar(x + (j - 1) * width, sub["expectation_ratio"], width,
                    yerr=sub["expectation_ratio_sd"], capsize=3, label=m,
                    color=PALETTE[m])
    for ax, ylab, title in [(axes[0], "Evaluations to target",
                             "Query cost under shot noise"),
                            (axes[1], r"$F_G(\theta)/C_{\max}$",
                             "Solution quality under shot noise")]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"S={s}" for s in shots])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0.5, 1.0)
    fig.tight_layout()
    return fig


def fig_seed_stability(src: Path) -> plt.Figure:
    """Evaluations to target per training/search seed for every method."""
    df = pd.read_csv(src / "EDFig3_SeedStability.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    seeds = list(df["seed"].unique())
    methods = [m for m in PALETTE if m in set(df["method"])]
    for m in methods:
        sub = df[df["method"] == m]
        ax.plot(range(len(seeds)), sub["evaluations"], marker="o",
                color=PALETTE[m], label=m,
                linewidth=2.4 if m == HIGHLIGHT else 1.4)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
    ax.set_xlabel("seed")
    ax.set_ylabel("Objective evaluations to target")
    ax.set_yscale("log")
    ax.set_title("Seed stability of the query benchmark")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_ablation(src: Path) -> plt.Figure:
    """Component ablation: evaluations to target per variant, 1-sd error bars."""
    df = pd.read_csv(src / "Figure5_Ablation.csv")
    meta = _meta(src)
    n = meta["config"]["train_n"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = range(len(df))
    colors = [PALETTE[HIGHLIGHT] if v == "Full method" else "#6baed6"
              for v in df["variant"]]
    ax.bar(x, df["evaluations"], yerr=df["evaluations_sd"], color=colors,
           capsize=4, edgecolor="black", linewidth=0.6)
    for xi, val in zip(x, df["evaluations"]):
        ax.text(xi, val + 4, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax.set_ylabel("Objective (circuit) evaluations to target")
    ax.set_title(rf"Component ablation at $n={n}$, $p=2$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_quality_cost(src: Path) -> plt.Figure:
    """Quality-cost plane: expectation ratio vs evaluations, with error bars."""
    df = pd.read_csv(src / "Figure2_QueryEfficiency.csv")
    meta = _meta(src)
    n = meta["config"]["train_n"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for _, row in df.iterrows():
        m = row["method"]
        ax.errorbar(row["evaluations"], row["expectation_ratio"],
                    xerr=row["evaluations_sd"], yerr=row["expectation_ratio_sd"],
                    marker="o", markersize=9, color=PALETTE.get(m, "#777777"),
                    capsize=3, linestyle="none")
        ax.annotate(m, (row["evaluations"], row["expectation_ratio"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Mean objective (circuit) evaluations to target")
    ax.set_ylabel(r"Expectation ratio $F_G(\theta)/C_{\max}$")
    ax.set_title(rf"Quality--cost plane at $n={n}$ (cost-to-target protocol)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_landscape(src: Path) -> plt.Figure:
    """Real 2D landscape slice with the predicted trust-region cross-section."""
    meta = _meta(src)
    ls = meta["landscape"]
    G = np.array(ls["gammas"]); Bv = np.array(ls["betas"])
    V = np.array(ls["values"])
    mu = ls["mu"]; sd = ls["sd"]; rad = ls["radius_scale"]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.contourf(G, Bv, V, levels=24, cmap="viridis")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$F_G(\theta)/C_{\max}$")
    # trust-region cross-section: axis-aligned ellipse (diagonal covariance)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mu[0] + rad * sd[0] * np.cos(t), mu[1] + rad * sd[1] * np.sin(t),
            color="white", lw=2.0, label="Trust region (2 s.d.)")
    ax.plot([mu[0]], [mu[1]], marker="*", markersize=14, color="white",
            markeredgecolor="black", linestyle="none", label="Predicted mean")
    ax.set_xlim(float(G.min()), float(G.max()))
    ax.set_ylim(float(Bv.min()), float(Bv.max()))
    ax.set_xlabel(r"$\gamma_1$")
    ax.set_ylabel(r"$\beta_1$")
    inst = ls["instance"]
    ax.set_title(rf"Landscape slice, {inst['family'].upper()} graph, "
                 rf"$n={inst['n']}$ (exact $p=2$ expectation)")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_pipeline(src: Path) -> plt.Figure:
    """Conceptual pipeline drawn from a REAL test instance and its prediction."""
    meta = _meta(src)
    ex = meta["pipeline_example"]
    import networkx as nx
    g = nx.Graph(); g.add_nodes_from(range(ex["n"])); g.add_edges_from(ex["edges"])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    # Panel 1: the input graph
    ax = axes[0]
    pos = nx.spring_layout(g, seed=3)
    nx.draw_networkx(g, pos=pos, ax=ax, node_size=140, node_color="#4292c6",
                     edge_color="0.5", with_labels=False)
    ax.set_title(f"(a) Input graph ({ex['family'].upper()}, $n={ex['n']}$)")
    ax.axis("off")
    # Panel 2: predicted Gaussian over angles (first two coordinates)
    ax = axes[1]
    mu = ex["mu"]; sd = ex["sd"]
    t = np.linspace(0, 2 * np.pi, 200)
    for k, alpha in [(1, 0.9), (2, 0.5)]:
        ax.plot(mu[0] + k * sd[0] * np.cos(t), mu[1] + k * sd[1] * np.sin(t),
                color=PALETTE[HIGHLIGHT], alpha=alpha, lw=1.8)
    ax.plot([mu[0]], [mu[1]], marker="*", markersize=13,
            color=PALETTE[HIGHLIGHT], linestyle="none")
    ax.set_xlabel(r"$\gamma_1$"); ax.set_ylabel(r"$\beta_1$")
    ax.set_title(r"(b) Predicted $\mathcal{N}(\mu(G),\Sigma(G))$")
    ax.spines[["top", "right"]].set_visible(False)
    # Panel 3: policy roles
    ax = axes[2]
    ax.axis("off")
    roles = [(r"$\mu(G)$", "initializes local search"),
             (r"$\Sigma(G)$", "preconditions steps and" "\nbounds the trust region"),
             ("stopping", "early, at the shared" "\nquality target")]
    for i, (sym, desc) in enumerate(roles):
        y = 0.88 - 0.33 * i
        ax.text(0.02, y, sym, fontsize=12, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.40, y, desc, fontsize=10.5, transform=ax.transAxes, va="top")
    ax.set_title("(c) Search policy")
    fig.tight_layout()
    return fig


FIGURE_BUILDERS = {
    "pipeline": ("Figure1_ConceptualPipeline", fig_pipeline),
    "query_efficiency": ("Figure2_QueryEfficiency", fig_query_efficiency),
    "calibration": ("Figure3_CalibrationAndUncertainty", fig_calibration),
    "generalization": ("Figure4_Generalization", fig_generalization),
    "ablation": ("Figure5_Ablation", fig_ablation),
    "quality_cost": ("Figure6_QualityCostFrontier", fig_quality_cost),
    "budget_policy": ("Figure7_UncertaintyBudgetPolicy", fig_budget_policy),
    "landscape": ("Figure9_LandscapeSlice", fig_landscape),
    "shot_noise": ("ExtendedData/ED_Fig2_ShotNoiseRobustness", fig_shot_noise),
    "seed_stability": ("ExtendedData/ED_Fig3_SeedStability", fig_seed_stability),
}


def build_all(source_data_dir, figures_dir, formats=("pdf", "png"),
              dpi=300) -> List[str]:
    """Render every data figure from the committed source data. Returns paths.

    Figures 8, 10 and 11 are conceptual schematics (theory map, repository
    architecture, reproducibility workflow) and are the only manuscript
    figures not rebuilt from source data.
    """
    _style()
    src = Path(source_data_dir)
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, (stem, builder) in FIGURE_BUILDERS.items():
        fig = builder(src)
        for ext in formats:
            p = out / f"{stem}.{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            written.append(str(p))
        plt.close(fig)
    return written
