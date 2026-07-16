"""Figure builders. Each function reads only the committed source-data files
(CSV + meta.json) written by the simulation pipeline, and returns a matplotlib
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
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


def _legacy_stopping_counts(src: Path) -> bool:
    return _meta(src).get("score_semantics") == \
        "recorded_objective_calls_until_target_convergence_or_cap"


def _cost_label(src: Path) -> str:
    if _legacy_stopping_counts(src):
        return "Recorded objective calls to termination"
    return "Capped objective-call cost to target"


def _style():
    plt.rcParams.update({
        "font.size": 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "savefig.bbox": "tight",
    })


def _diagram_box(ax, xy, width, height, text, *, face="#f7f9fb",
                 edge="#263238", fontsize=10.5, weight="normal"):
    """Add a publication-safe rounded box in axes coordinates."""
    patch = FancyBboxPatch(
        xy, width, height, transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, linespacing=1.25)
    return patch


def _diagram_arrow(ax, start, end, *, color="#455a64"):
    """Add a straight, non-crossing arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=12, linewidth=1.25, color=color,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)
    return arrow


def fig_query_efficiency(src: Path) -> plt.Figure:
    """Bar chart with 1-sd error bars for the recorded cost statistic."""
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
    ax.set_ylabel(_cost_label(src))
    if _legacy_stopping_counts(src):
        title = rf"Recorded optimizer calls at $n={n}$, $p=2$"
    else:
        title = rf"Capped query cost to the $98\%$ target at $n={n}$, $p=2$"
    ax.set_title(title + rf" ({n_test} test instances)")
    ax.set_ylim(bottom=0)
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
    """Cross-size transfer of the recorded cost statistic."""
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
    ax.set_ylabel(_cost_label(src))
    ax.set_yscale("log")
    ax.set_title(rf"Cross-size transfer (trained at $n={n_train}$)")
    ax.set_xticks(df["n"])
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_budget_policy(src: Path) -> plt.Figure:
    """Difficulty-score-to-budget rule with realized test-set allocations."""
    df = pd.read_csv(src / "Figure7_BudgetPolicy.csv")
    meta = _meta(src)
    budget = meta["config"]["budget"]
    cap_min = meta["config"].get("budget_cap_min", 40)
    zg = np.linspace(-3, 3, 301)
    K_rule = np.clip(np.floor(1 + 4 / (1 + np.exp(-zg))), 1, 4)
    T_rule = np.clip(np.floor((budget // 2) * (0.5 + np.maximum(zg, 0))),
                     cap_min, budget)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    ax.step(zg, K_rule, where="post", color=PALETTE[HIGHLIGHT], lw=2)
    ax.plot(df["z"], df["K_seeds"], "o", color="#444", ms=5, alpha=0.7,
            label="test instances")
    ax.set_xlabel("standardized difficulty score $z$")
    ax.set_ylabel("initial seed points $K$")
    ax.set_title("Seed allocation $K(z)$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(zg, T_rule, color=PALETTE[HIGHLIGHT], lw=2)
    ax.plot(df["z"], df["T_budget_cap"], "o", color="#444", ms=5, alpha=0.7,
            label="test instances")
    ax.set_xlabel("standardized difficulty score $z$")
    ax.set_ylabel("evaluation cap $T$")
    ax.set_title("Budget cap $T(z)$")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validation-fit difficulty sets the per-instance search budget",
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
    for ax, ylab, title in [(axes[0], _cost_label(src),
                             "Recorded query count under shot noise"),
                            (axes[1], r"$F_G(\theta)/C_{\max}$",
                             "Solution quality under shot noise")]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"S={s}" for s in shots])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0.5, 1.0)
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def fig_seed_stability(src: Path) -> plt.Figure:
    """Recorded cost statistic per training/search seed and method."""
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
    ax.set_ylabel(_cost_label(src))
    ax.set_yscale("log")
    ax.set_title("Seed stability of the query benchmark")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_ablation(src: Path) -> plt.Figure:
    """Component ablation of the recorded cost statistic."""
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
    ax.set_ylabel(_cost_label(src))
    ax.set_title(rf"Component ablation at $n={n}$, $p=2$")
    ax.set_ylim(bottom=0)
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
        offsets = {
            "Heuristic": (-8, -14),
            "GCTR": (8, 10),
            "k-NN": (8, -12),
            "TQA": (8, -8),
        }
        dx, dy = offsets.get(m, (8, 4))
        ax.annotate(m, (row["evaluations"], row["expectation_ratio"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9,
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("Mean " + _cost_label(src).lower())
    ax.set_ylabel(r"Expectation ratio $F_G(\theta)/C_{\max}$")
    suffix = ("historical stopping-count protocol"
              if _legacy_stopping_counts(src)
              else "capped cost-to-target protocol")
    ax.set_title(rf"Quality--cost plane at $n={n}$ ({suffix})")
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_landscape(src: Path) -> plt.Figure:
    """2D landscape slice with the predicted trust-region cross-section."""
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
    """Conceptual pipeline drawn from one test instance and its prediction."""
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
             (r"$\widehat e(G)$", "sets nominal seeds and" "\nan early cap")]
    for i, (sym, desc) in enumerate(roles):
        y = 0.88 - 0.33 * i
        ax.text(0.02, y, sym, fontsize=12, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.40, y, desc, fontsize=10.5, transform=ax.transAxes, va="top")
    ax.set_title("(c) Search policy")
    fig.tight_layout()
    return fig


def fig_theory_map(src: Path) -> plt.Figure:
    """Map implemented policy components to proofs and measured controls."""
    del src
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.set_axis_off()
    ax.set_title("Conditional theory and empirical audit", pad=14,
                 fontweight="semibold")

    _diagram_box(ax, (0.02, 0.35), 0.16, 0.25,
                 "Input graph $G$\nGIN + spectral\nfeatures",
                 face="#edf4fb", fontsize=9.3, weight="semibold")
    _diagram_box(ax, (0.24, 0.60), 0.18, 0.20,
                 r"Gaussian head" "\n" r"$\mu(G),\,\Sigma(G)$", face="#fff5f0")
    _diagram_box(ax, (0.24, 0.18), 0.18, 0.20,
                 r"Validation-fit score" "\n" r"$\widehat e(G)$", face="#f3eef8")
    _diagram_box(ax, (0.49, 0.60), 0.20, 0.20,
                 "Search geometry\npreconditioned steps\nMahalanobis boundary",
                 face="#fff5f0", fontsize=9.8)
    _diagram_box(ax, (0.49, 0.18), 0.20, 0.20,
                 "Resource rule\nnominal seeds $K$\nearly cap $T$",
                 face="#f3eef8", fontsize=9.8)
    _diagram_box(ax, (0.76, 0.56), 0.21, 0.25,
                 "Conditional statements\nlocal objective retention\nshot-estimation bounds",
                 face="#eef7ee", fontsize=9.8)
    _diagram_box(ax, (0.76, 0.14), 0.21, 0.25,
                 "Audited evidence\ncapped cost + attainment\npaired controls",
                 face="#fff8e5", fontsize=9.8, weight="semibold")

    _diagram_arrow(ax, (0.18, 0.52), (0.24, 0.70))
    _diagram_arrow(ax, (0.18, 0.43), (0.24, 0.28))
    _diagram_arrow(ax, (0.42, 0.70), (0.49, 0.70))
    _diagram_arrow(ax, (0.42, 0.28), (0.49, 0.28))
    _diagram_arrow(ax, (0.69, 0.70), (0.76, 0.69))
    _diagram_arrow(ax, (0.69, 0.28), (0.76, 0.27))
    ax.text(0.5, 0.02,
            "Bounds are conditional; controls determine whether the full policy improves cost.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.4,
            color="#37474f")
    fig.tight_layout()
    return fig


def fig_repository_architecture(src: Path) -> plt.Figure:
    """Reviewer-facing package layout using only paths that ship."""
    del src
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_axis_off()
    ax.set_title("Review package architecture", pad=14,
                 fontweight="semibold")

    rows = [
        (0.68, "Installable source\ncode/specops_gctr/",
         "Public API + CLIs\ngctr-optimize\ngctr-reproduce",
         "Wheel / editable install"),
        (0.40, "Release tests\ncode/tests/",
         "31 unit + artifact tests\nmath, accounting, hashes",
         "Validated package"),
        (0.12, "Committed evidence\nmanuscript/source_data/",
         "Data figures + tables\nportable manifest",
         "main.tex / main.pdf\narXiv source archive"),
    ]
    colors = ["#edf4fb", "#eef7ee", "#fff8e5"]
    for (y, left, middle, right), color in zip(rows, colors):
        _diagram_box(ax, (0.03, y), 0.25, 0.17, left, face=color,
                     fontsize=9.8, weight="semibold")
        _diagram_box(ax, (0.37, y), 0.27, 0.17, middle, face="#f7f9fb",
                     fontsize=9.6)
        _diagram_box(ax, (0.73, y), 0.24, 0.17, right, face=color,
                     fontsize=9.6)
        _diagram_arrow(ax, (0.28, y + 0.085), (0.37, y + 0.085))
        _diagram_arrow(ax, (0.64, y + 0.085), (0.73, y + 0.085))
    ax.text(0.5, 0.01,
            "Generator source is archived separately from display-only plotting and table code.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.2,
            color="#37474f")
    fig.tight_layout()
    return fig


def fig_reproducibility_workflow(src: Path) -> plt.Figure:
    """End-to-end workflow with a non-crossing, auditable execution order."""
    del src
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.set_axis_off()
    ax.set_title("End-to-end reproducibility workflow", pad=14,
                 fontweight="semibold")

    top = [
        (0.03, "1  Generate graphs\n+ privileged labels"),
        (0.28, "2  Fit predictor\n+ validation score"),
        (0.53, "3  Evaluate methods\n+ controls"),
        (0.78, "4  Record instance rows\n+ environment"),
    ]
    for x, label in top:
        _diagram_box(ax, (x, 0.62), 0.19, 0.17, label,
                     face="#edf4fb", fontsize=9.4,
                     weight="semibold" if x == 0.03 else "normal")
    for i in range(3):
        _diagram_arrow(ax, (top[i][0] + 0.19, 0.705),
                       (top[i + 1][0], 0.705))

    bottom = [
        (0.78, "5  Aggregate statistics\n+ paired tests"),
        (0.53, "6  Render figures\n+ generated tables"),
        (0.28, "7  Validate tests\n+ portable hashes"),
        (0.03, "8  Build wheel, PDF\n+ arXiv archive"),
    ]
    for x, label in bottom:
        _diagram_box(ax, (x, 0.24), 0.19, 0.17, label,
                     face="#eef7ee", fontsize=9.4,
                     weight="semibold" if x == 0.03 else "normal")
    _diagram_arrow(ax, (0.875, 0.62), (0.875, 0.41))
    for i in range(3):
        _diagram_arrow(ax, (bottom[i][0], 0.325),
                       (bottom[i + 1][0] + 0.19, 0.325))

    ax.text(0.5, 0.04,
            "Full experiment  •  replot-only  •  validate-only",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.4,
            color="#37474f")
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
    "theory_map": ("Figure8_TheoryMap", fig_theory_map),
    "landscape": ("Figure9_LandscapeSlice", fig_landscape),
    "repository_architecture": ("Figure10_RepositoryArchitecture",
                                fig_repository_architecture),
    "reproducibility_workflow": ("Figure11_ReproducibilityWorkflow",
                                 fig_reproducibility_workflow),
    "shot_noise": ("ExtendedData/ED_Fig2_ShotNoiseRobustness", fig_shot_noise),
    "seed_stability": ("ExtendedData/ED_Fig3_SeedStability", fig_seed_stability),
}


def build_all(source_data_dir, figures_dir, formats=("pdf", "png"),
              dpi=300) -> List[str]:
    """Render every data figure and deterministic schematic. Returns paths."""
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
            kwargs = {"dpi": dpi, "bbox_inches": "tight"}
            if ext.lower() == "pdf":
                # Matplotlib otherwise inserts the wall-clock time into every
                # PDF, making a source-identical replot fail byte-level checks.
                kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
            fig.savefig(p, **kwargs)
            written.append(str(p))
        plt.close(fig)
    return written
