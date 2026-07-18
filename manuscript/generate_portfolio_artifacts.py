#!/usr/bin/env python3
"""Generate the portfolio manuscript figures, tables and numerical macros.

The script reads only the validated, locked portfolio evidence.  It performs
no optimizer fitting and does not modify any experimental record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
FIGURES = MANUSCRIPT / "figures"
TABLES = MANUSCRIPT / "tables"
sys.path.insert(0, str(ROOT))

from specops_gctr.portfolio_experiment import validate_portfolio_manifest  # noqa: E402
from specops_gctr.portfolio import GRAPH_FEATURE_NAMES  # noqa: E402
from specops_gctr.risk_control import summarize_locked_audit  # noqa: E402


STUDIES = {
    "regular_development": (
        ROOT / "configs/portfolio_development.json",
        ROOT / "portfolio_results/development_v2_tqa_family_gate",
    ),
    "heterogeneous_development": (
        ROOT / "configs/portfolio_heterogeneous_development.json",
        ROOT / "portfolio_results/heterogeneous_development_v2_tqa_family_gate",
    ),
    "confirmatory": (
        ROOT / "configs/portfolio_heterogeneous_confirmatory.json",
        ROOT / "portfolio_results/heterogeneous_confirmatory_v1",
    ),
}

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E69F00",
    "green": "#3A7D44",
    "red": "#B33B3B",
    "purple": "#775DA6",
    "gray": "#6B7280",
    "light": "#E8EEF3",
    "ink": "#20252B",
}
FAMILY_ORDER = ("er", "rr", "ba", "ws")
FAMILY_LABEL = {
    "er": "Erdos-Renyi",
    "rr": "random regular",
    "ba": "Barabasi-Albert",
    "ws": "Watts-Strogatz",
}
FAMILY_TEX_LABEL = {
    "er": r"Erd\H{o}s--R\'enyi",
    "rr": "random regular",
    "ba": r"Barab\'asi--Albert",
    "ws": "Watts--Strogatz",
}
FAMILY_COLOR = {
    "er": COLORS["gray"],
    "rr": COLORS["blue"],
    "ba": COLORS["red"],
    "ws": COLORS["green"],
}
ARM_LABEL = {
    "concentration": "concentration",
    "tqa_dt_0p25": "TQA 0.25",
    "tqa_dt_0p50": "TQA 0.50",
    "tqa_dt_0p75": "TQA 0.75",
    "tqa_dt_1p00": "TQA 1.00",
    "knn": "1-NN transfer",
    "random_multistart": "random multistart",
    "spsa": "SPSA",
    "legacy_gctr": "legacy GCTR",
}
FIGURE_STEMS = (
    "Figure1_PortfolioProtocol",
    "Figure2_ConfirmatoryAudit",
    "Figure3_DevelopmentToConfirmation",
    "Figure4_ArmOpportunityRanking",
    "Figure5_ResidualOrderStatistic",
    "Figure6_PredictedVsRealizedRoutes",
    "Figure7_AnytimeByFamily",
    "Figure8_FamilyEffects",
    "Figure9_RiskSensitivity",
    "Figure10_ArmTransitions",
    "Figure11_SplitDiagnostics",
    "Figure12_ResourceLedger",
    "Figure13_InferenceThresholdSensitivity",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence() -> dict:
    evidence = {}
    for name, (config_path, result_dir) in STUDIES.items():
        config = _load_json(config_path)
        validate_portfolio_manifest(result_dir, config)
        evidence[name] = {
            "config": config,
            "result_dir": result_dir,
            "summary": _load_json(result_dir / "summary.json"),
            "decision": _load_json(result_dir / "frozen_decision.json"),
            "manifest": _load_json(result_dir / "portfolio_manifest.json"),
            "traces": _load_jsonl(result_dir / "traces.jsonl"),
        }
    return evidence


def _split_trace_matrix(item: dict, split: str) -> dict:
    """Return one validated graph-by-arm matrix from committed trace rows."""
    arms = tuple(item["summary"]["arms"])
    rows = [row for row in item["traces"] if row["split"] == split]
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        graph_rows = grouped.setdefault(row["graph_id"], {})
        if row["arm"] in graph_rows:
            raise ValueError(
                f"duplicate {split} trace for {row['graph_id']}:{row['arm']}")
        graph_rows[row["arm"]] = row
    graph_ids = tuple(sorted(grouped))
    if not graph_ids:
        raise ValueError(f"no trace rows found for split {split!r}")

    features = []
    utilities = []
    metadata = []
    for graph_id in graph_ids:
        by_arm = grouped[graph_id]
        if set(by_arm) != set(arms):
            raise ValueError(
                f"{split} graph {graph_id} does not contain every frozen arm")
        first = by_arm[arms[0]]
        reference_features = np.asarray(first["graph_features"], dtype=float)
        for arm in arms[1:]:
            candidate = np.asarray(by_arm[arm]["graph_features"], dtype=float)
            if not np.array_equal(candidate, reference_features):
                raise ValueError(
                    f"graph features vary across arms for {graph_id}")
        features.append(reference_features)
        utilities.append([float(by_arm[arm]["aurc"]) for arm in arms])
        metadata.append({
            "graph_id": graph_id,
            "family": first["family"],
            "n": int(first["n"]),
            "generator_parameters": first["generator_parameters"],
        })
    return {
        "arms": arms,
        "graph_ids": graph_ids,
        "features": np.asarray(features, dtype=float),
        "utilities": np.asarray(utilities, dtype=float),
        "metadata": metadata,
    }


def _routing_diagnostics(item: dict, split: str) -> dict:
    """Recompute routing quantities from the frozen selector and trace matrix."""
    matrix = _split_trace_matrix(item, split)
    arms = matrix["arms"]
    arm_index = {arm: index for index, arm in enumerate(arms)}
    selector = item["decision"]["selector"]
    feature_mean = np.asarray(selector["feature_mean"], dtype=float)
    feature_scale = np.asarray(selector["feature_scale"], dtype=float)
    coefficients = np.asarray(selector["coefficients"], dtype=float)
    if coefficients.shape != (matrix["features"].shape[1] + 1, len(arms)):
        raise ValueError("frozen selector coefficient shape is inconsistent")
    standardized = (matrix["features"] - feature_mean) / feature_scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    predictions = design @ coefficients
    selected_indices = np.argmin(predictions, axis=1)
    selected_arms = np.asarray(
        [arms[index] for index in selected_indices], dtype=object)
    fallback_arms = np.asarray([
        item["decision"]["family_control_arm_by_family"].get(
            row["family"], item["decision"]["baseline_arm"])
        for row in matrix["metadata"]
    ], dtype=object)
    fallback_indices = np.asarray(
        [arm_index[arm] for arm in fallback_arms], dtype=int)
    row_indices = np.arange(len(matrix["graph_ids"]))
    predicted_delta = (
        predictions[row_indices, selected_indices]
        - predictions[row_indices, fallback_indices]
    )
    selected_delta = (
        matrix["utilities"][row_indices, selected_indices]
        - matrix["utilities"][row_indices, fallback_indices]
    )
    residuals = selected_delta - predicted_delta
    abstainer = item["decision"]["abstainer"]
    quantile = float(abstainer["residual_quantile"])
    upper_bounds = predicted_delta + quantile
    accepted = (
        (selected_arms != fallback_arms)
        & (upper_bounds < -float(abstainer["margin"]))
    )
    deployed_arms = np.where(accepted, selected_arms, fallback_arms)
    deployed_delta = np.where(accepted, selected_delta, 0.0)
    covered = residuals <= quantile + 1e-15

    result = {
        **matrix,
        "predictions": predictions,
        "selected_indices": selected_indices,
        "selected_arms": selected_arms,
        "fallback_indices": fallback_indices,
        "fallback_arms": fallback_arms,
        "predicted_delta": predicted_delta,
        "selected_delta": selected_delta,
        "residuals": residuals,
        "quantile": quantile,
        "upper_bounds": upper_bounds,
        "accepted": accepted,
        "deployed_arms": deployed_arms,
        "deployed_delta": deployed_delta,
        "covered": covered,
    }
    if split == "audit":
        committed = {
            row["graph_id"]: row
            for row in item["summary"]["splits"]["audit"]["per_graph"]
        }
        for index, graph_id in enumerate(matrix["graph_ids"]):
            row = committed[graph_id]
            if row["selected_arm"] != selected_arms[index]:
                raise ValueError("recomputed selected arm differs from summary")
            if row["deployed_arm"] != deployed_arms[index]:
                raise ValueError("recomputed deployed arm differs from summary")
            if bool(row["accepted"]) != bool(accepted[index]):
                raise ValueError("recomputed acceptance differs from summary")
            if not np.isclose(
                    row["upper_bound"], upper_bounds[index],
                    rtol=0.0, atol=1e-12):
                raise ValueError("recomputed upper bound differs from summary")
            if not np.isclose(
                    row["delta_vs_family_control"], deployed_delta[index],
                    rtol=0.0, atol=1e-12):
                raise ValueError("recomputed deployed delta differs from summary")
    return result


def _stratified_bootstrap_interval(
    values: np.ndarray,
    groups: list[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Vectorized deterministic stratified percentile interval."""
    data = np.asarray(values, dtype=float)
    labels = np.asarray(groups, dtype=object)
    if data.ndim != 1 or labels.shape != data.shape:
        raise ValueError("bootstrap values and groups must be aligned vectors")
    rng = np.random.default_rng(int(seed))
    means = np.zeros(int(resamples), dtype=float)
    for label in sorted(set(groups)):
        stratum = data[labels == label]
        draws = rng.integers(
            0, stratum.size, size=(int(resamples), stratum.size))
        means += stratum[draws].sum(axis=1)
    means /= data.size
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    metadata = {
        "Title": stem,
        "Author": "Molena Huynh",
        "Creator": "generate_portfolio_artifacts.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, metadata=metadata)
    fig.savefig(png, dpi=300, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return [pdf, png]


def _box(ax, xy, width, height, title, body, color):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=1.1, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + height - 0.032, title, weight="bold",
            color=color, va="top", fontsize=7.8)
    ax.text(x + 0.014, y + height - 0.079, body, va="top",
            fontsize=6.6, linespacing=1.22)


def make_protocol_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    summary = item["summary"]
    decision = item["decision"]
    config = item["config"]
    audit = summary["splits"]["audit"]
    q = decision["abstainer"]["residual_quantile"]
    hashes = item["manifest"]

    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0, 0.985, "An internally locked, target-free decision path",
            fontsize=12, weight="bold", va="top")

    xs = [0.01, 0.207, 0.404, 0.601, 0.798]
    titles = [
        "1  Freeze",
        "2  Trace",
        "3  Route",
        "4  Calibrate",
        "5  Audit once",
    ]
    bodies = [
        "48 label graphs\n48 development\n48 calibration",
        "9 fixed arms\n32 calls per arm\n6 checkpoints",
        "16 graph features\nridge predictor\nfamily fallback",
        f"residual rank rule\nq = {q:.5f}\nalpha = {config['selector']['alpha']:.1f}",
        f"{audit['n_graphs']} new graphs\n20 per stratum\nno target stop",
    ]
    box_colors = [COLORS["gray"], COLORS["blue"], COLORS["purple"],
                  COLORS["orange"], COLORS["red"]]
    for x, title, body, color in zip(xs, titles, bodies, box_colors):
        _box(ax, (x, 0.55), 0.176, 0.28, title, body, color)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 0.197, 0.69), xytext=(x + 0.177, 0.69),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0,
                                "color": COLORS["gray"]})

    resources = summary["resource_accounting"]
    resource_text = (
        "Exact objective-query ledger: "
        f"labels {resources['angle_label_objective_queries']:,}  |  "
        f"development {resources['trace_objective_queries_by_split']['development']:,}  |  "
        f"calibration {resources['trace_objective_queries_by_split']['calibration']:,}  |  "
        f"audit {resources['trace_objective_queries_by_split']['audit']:,}  |  "
        "device shots 0"
    )
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.375), 0.96, 0.085,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.9, edgecolor=COLORS["gray"], facecolor=COLORS["light"],
    ))
    ax.text(0.49, 0.417, resource_text, ha="center", va="center",
            fontsize=6.8, weight="bold")

    gates = audit["go_no_go"]
    gate_text = (
        "Locked decision: 4,000-draw bootstrap criterion passes\n"
        f"empirical coverage {audit['empirical_one_sided_coverage']:.3f} "
        f"< {1.0-config['selector']['alpha']:.2f}; reliability gate FAILS"
    )
    ax.add_patch(FancyBboxPatch(
        (0.12, 0.17), 0.76, 0.125,
        boxstyle="round,pad=0.015,rounding_size=0.012",
        linewidth=1.4, edgecolor=COLORS["red"], facecolor="#FBECEC",
    ))
    ax.text(0.5, 0.232, gate_text, ha="center", va="center", fontsize=7.8,
            color=COLORS["red"], weight="bold", linespacing=1.25)
    assert not gates["nmi_claim_ready"]

    ax.text(
        0.5, 0.075,
        "SHA-256  implementation " + hashes["implementation_sha256"][:12]
        + "...  |  protocol " + hashes["protocol_sha256"][:12]
        + "...  |  decision " + hashes["decision_sha256"][:12] + "...",
        ha="center", va="center", fontsize=7, family="monospace",
        color=COLORS["gray"],
    )
    return save_figure(fig, "Figure1_PortfolioProtocol")


def _confirmatory_curve_data(item: dict):
    audit = item["summary"]["splits"]["audit"]
    records = [row for row in item["traces"] if row["split"] == "audit"]
    trace_by_pair = {(row["graph_id"], row["arm"]): row for row in records}
    graph_rows = audit["per_graph"]
    checkpoints = np.asarray(item["summary"]["checkpoints"], dtype=int)
    curves = {name: [] for name in ("global", "family", "deployed", "oracle")}
    global_arm = audit["baseline_arm"]
    arms = item["summary"]["arms"]
    for graph in graph_rows:
        gid = graph["graph_id"]
        global_ratio = np.asarray(
            trace_by_pair[(gid, global_arm)]["expectation_ratios"], dtype=float)
        family_ratio = np.asarray(
            trace_by_pair[(gid, graph["family_control_arm"])]["expectation_ratios"],
            dtype=float,
        )
        deployed_ratio = np.asarray(
            trace_by_pair[(gid, graph["deployed_arm"])]["expectation_ratios"],
            dtype=float,
        )
        all_ratios = np.stack([
            trace_by_pair[(gid, arm)]["expectation_ratios"] for arm in arms])
        curves["global"].append(1.0 - global_ratio)
        curves["family"].append(1.0 - family_ratio)
        curves["deployed"].append(1.0 - deployed_ratio)
        curves["oracle"].append(1.0 - all_ratios.max(axis=0))
    return checkpoints, {key: np.asarray(value) for key, value in curves.items()}


def make_confirmatory_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    audit = item["summary"]["splits"]["audit"]
    config = item["config"]
    checkpoints, curves = _confirmatory_curve_data(item)
    fig = plt.figure(figsize=(7.25, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

    ax = fig.add_subplot(grid[0, 0])
    for xpos, family in enumerate(FAMILY_ORDER):
        accepted = [row for row in audit["per_graph"]
                    if row["family"] == family and row["accepted"]]
        deltas = np.asarray([row["delta_vs_family_control"]
                             for row in accepted], dtype=float)
        if deltas.size:
            jitter = np.linspace(-0.10, 0.10, deltas.size)
            colors = [COLORS["red"] if value > 0 else COLORS["blue"]
                      for value in deltas]
            ax.scatter(xpos + jitter, deltas, c=colors, s=34,
                       edgecolor="white", linewidth=0.5, zorder=3)
        harms = sum(value > 0 for value in deltas)
        ax.text(xpos, 0.0135, f"{deltas.size} accepted\n{harms} harmed",
                ha="center", va="top", fontsize=7)
    ax.axhline(0, color=COLORS["ink"], lw=0.9)
    ax.set_xticks(range(4), ["ER", "RR", "BA", "WS"])
    ax.set_ylabel("AURC difference vs family fallback")
    ax.set_ylim(-0.082, 0.018)
    ax.set_title("a  Only 12 deployments were accepted", loc="left",
                 weight="bold")

    ax = fig.add_subplot(grid[0, 1])
    curve_styles = [
        ("global", "global TQA dt=0.75", COLORS["gray"], "--"),
        ("family", "family fallback", COLORS["orange"], "-"),
        ("deployed", "abstaining deployment", COLORS["blue"], "-"),
        ("oracle", "per-checkpoint oracle", COLORS["green"], ":"),
    ]
    for key, label, color, linestyle in curve_styles:
        mean = curves[key].mean(axis=0)
        ax.plot(checkpoints, mean, marker="o", ms=3.5, lw=1.6,
                color=color, linestyle=linestyle, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(checkpoints, [str(value) for value in checkpoints])
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("mean normalized regret")
    ax.set_title("b  Target-free anytime performance", loc="left",
                 weight="bold")
    ax.grid(axis="y", color="#D5D9DD", lw=0.6)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(len(FAMILY_ORDER))
    width = 0.36
    family_values = [audit["per_family"][family][
        "family_label_control_mean_aurc"] for family in FAMILY_ORDER]
    deployed_values = [audit["per_family"][family][
        "gated_selector_mean_aurc"] for family in FAMILY_ORDER]
    ax.bar(x - width / 2, family_values, width, color=COLORS["orange"],
           label="family fallback")
    ax.bar(x + width / 2, deployed_values, width, color=COLORS["blue"],
           label="gated deployment")
    ax.set_xticks(x, ["ER", "RR", "BA", "WS"])
    ax.set_ylabel("mean AURC (lower is better)")
    ax.set_ylim(0.14, 0.20)
    ax.set_title("c  Mean gain is concentrated in RR", loc="left",
                 weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", color="#D5D9DD", lw=0.6)

    ax = fig.add_subplot(grid[1, 1])
    coverage = audit["empirical_one_sided_coverage"]
    nominal = 1.0 - float(config["selector"]["alpha"])
    joint_harm = audit["joint_harm_rate"]
    acceptance = audit["accepted_rate"]
    labels = ["coverage", "joint harm", "acceptance"]
    values = [coverage, joint_harm, acceptance]
    colors = [COLORS["red"], COLORS["green"], COLORS["blue"]]
    ax.barh(np.arange(3), values, color=colors, height=0.52)
    ax.axvline(nominal, color=COLORS["red"], linestyle="--", lw=1.1,
               label="coverage requirement 0.90")
    ax.axvline(float(config["selector"]["alpha"]), color=COLORS["green"],
               linestyle=":", lw=1.1, label="joint-harm ceiling 0.10")
    for ypos, value in enumerate(values):
        ax.text(min(value + 0.018, 0.96), ypos, f"{value:.3f}", va="center",
                fontsize=7.5, weight="bold")
    ax.set_yticks(np.arange(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("empirical fraction")
    ax.set_title("d  Descriptive coverage gate fails", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("One-shot confirmatory audit: a small average gain without reliable routing",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    return save_figure(fig, "Figure2_ConfirmatoryAudit")


def make_progression_figure(evidence: dict) -> list[Path]:
    names = ("regular_development", "heterogeneous_development", "confirmatory")
    labels = ("regular development\n(n=48)",
              "heterogeneous development\n(n=64)",
              "confirmatory\n(n=160)")
    audits = [evidence[name]["summary"]["splits"]["audit"] for name in names]
    configs = [evidence[name]["config"] for name in names]

    effects = np.asarray([
        item["mean_delta_gated_minus_family_label_control"] for item in audits])
    intervals = np.asarray([
        item["family_control_delta_bootstrap_95_interval"] for item in audits])
    coverage = np.asarray([
        item["empirical_one_sided_coverage"] for item in audits])
    accepted = np.asarray([item["accepted_rate"] for item in audits])
    harms = np.asarray([item["joint_harm_rate"] for item in audits])

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.25),
                             gridspec_kw={"wspace": 0.43})
    y = np.arange(3)
    xerr = np.vstack([effects - intervals[:, 0], intervals[:, 1] - effects])
    axes[0].errorbar(effects, y, xerr=xerr, fmt="o", color=COLORS["blue"],
                     ecolor=COLORS["blue"], capsize=3, lw=1.4)
    axes[0].axvline(0, color=COLORS["ink"], lw=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("gated - family AURC")
    axes[0].set_title("a  Effect and 95% interval", loc="left", weight="bold")
    axes[0].grid(axis="x", color="#D5D9DD", lw=0.6)

    bar_colors = [COLORS["red"] if value < 1.0-config["selector"]["alpha"]
                  else COLORS["green"]
                  for value, config in zip(coverage, configs)]
    axes[1].bar(y, coverage, color=bar_colors, width=0.58)
    axes[1].axhline(0.90, color=COLORS["red"], linestyle="--", lw=1.0)
    axes[1].set_xticks(y, ["regular", "hetero", "confirm"])
    axes[1].set_ylim(0.70, 1.0)
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("b  Descriptive coverage gate", loc="left", weight="bold")
    for xpos, value in enumerate(coverage):
        axes[1].text(xpos, value + 0.008, f"{value:.3f}", ha="center",
                     fontsize=7.2)

    x = np.arange(3)
    width = 0.34
    axes[2].bar(x - width/2, accepted, width, color=COLORS["blue"],
                label="accepted")
    axes[2].bar(x + width/2, harms, width, color=COLORS["red"],
                label="joint harms")
    axes[2].axhline(0.10, color=COLORS["red"], linestyle=":", lw=1.0)
    axes[2].set_xticks(x, ["regular", "hetero", "confirm"])
    axes[2].set_ylabel("fraction of audit graphs")
    axes[2].set_ylim(0, 0.32)
    axes[2].set_title("c  Selectivity and harm", loc="left", weight="bold")
    axes[2].legend(frameon=False)

    fig.suptitle("Development selected the protocol; confirmation judged it",
                 x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure3_DevelopmentToConfirmation")


def make_arm_opportunity_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    arms = tuple(item["summary"]["arms"])
    splits = ("development", "calibration", "audit")
    split_labels = ("development", "calibration", "audit")
    split_colors = (COLORS["gray"], COLORS["orange"], COLORS["blue"])
    means = {
        split: np.asarray([
            item["summary"]["splits"][split]["mean_aurc_by_arm"][arm]
            for arm in arms
        ], dtype=float)
        for split in splits
    }
    audit_matrix = _split_trace_matrix(item, "audit")
    winners = np.argmin(audit_matrix["utilities"], axis=1)
    win_counts = np.bincount(winners, minlength=len(arms))
    audit_order = np.argsort(means["audit"])

    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 4.35),
        gridspec_kw={"width_ratios": [1.45, 0.85], "wspace": 0.38},
    )
    y = np.arange(len(arms))
    offsets = (-0.18, 0.0, 0.18)
    for split, label, color, offset in zip(
            splits, split_labels, split_colors, offsets):
        axes[0].scatter(
            means[split], y + offset, s=28, color=color, label=label,
            edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].set_yticks(y, [ARM_LABEL[arm] for arm in arms])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean AURC (lower is better)")
    axes[0].set_title(
        "a  Mean arm utility across frozen splits",
        loc="left", weight="bold")
    axes[0].grid(axis="x", color="#D5D9DD", lw=0.6)
    axes[0].legend(frameon=False, ncol=3, loc="lower right")

    ranked_arms = [arms[index] for index in audit_order]
    ranked_counts = win_counts[audit_order]
    bars = axes[1].barh(
        np.arange(len(arms)), ranked_counts,
        color=[
            COLORS["blue"] if count else COLORS["light"]
            for count in ranked_counts
        ],
    )
    axes[1].set_yticks(
        np.arange(len(arms)), [ARM_LABEL[arm] for arm in ranked_arms])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("audit graphs won")
    axes[1].set_title(
        "b  Per-graph portfolio opportunity",
        loc="left", weight="bold")
    axes[1].grid(axis="x", color="#D5D9DD", lw=0.6)
    for bar, value in zip(bars, ranked_counts):
        axes[1].text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)), va="center", fontsize=7)
    fig.suptitle(
        "Arm ranking is stable on average but heterogeneous by graph",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure4_ArmOpportunityRanking")


def make_residual_order_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    calibration = _routing_diagnostics(item, "calibration")
    audit = _routing_diagnostics(item, "audit")
    q = calibration["quantile"]
    alpha = float(item["decision"]["abstainer"]["alpha"])
    sorted_calibration = np.sort(calibration["residuals"])
    rank = int(np.ceil((sorted_calibration.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), sorted_calibration.size)
    if not np.isclose(
            sorted_calibration[rank - 1], q, rtol=0.0, atol=1e-12):
        raise ValueError("recomputed calibration order statistic is inconsistent")

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), constrained_layout=True)
    indices = np.arange(1, sorted_calibration.size + 1)
    axes[0].plot(
        indices, sorted_calibration, color=COLORS["blue"], marker="o",
        ms=2.8, lw=1.1)
    axes[0].scatter(
        [rank], [q], s=55, color=COLORS["red"], edgecolor="white",
        linewidth=0.6, zorder=4,
        label=f"rank {rank}/{sorted_calibration.size}")
    axes[0].axhline(
        q, color=COLORS["red"], linestyle="--", lw=1.0,
        label=f"q = {q:.5f}")
    axes[0].axhline(0, color=COLORS["ink"], lw=0.7)
    axes[0].set_xlabel("ordered calibration residual")
    axes[0].set_ylabel(r"$R=\Delta-\widehat{\Delta}$")
    axes[0].set_title(
        "a  Frozen one-sided order statistic", loc="left", weight="bold")
    axes[0].legend(frameon=False)

    for values, label, color, linestyle in (
        (calibration["residuals"], "calibration (n=48)",
         COLORS["blue"], "-"),
        (audit["residuals"], "audit (n=160; descriptive)",
         COLORS["orange"], "--"),
    ):
        ordered = np.sort(values)
        cdf = np.arange(1, ordered.size + 1) / ordered.size
        axes[1].step(
            ordered, cdf, where="post", color=color, linestyle=linestyle,
            lw=1.6, label=label)
    axes[1].axvline(q, color=COLORS["red"], linestyle=":", lw=1.1)
    axes[1].axhline(
        1.0 - alpha, color=COLORS["gray"], linestyle=":", lw=0.9)
    axes[1].set_xlabel("selected-minus-fallback residual")
    axes[1].set_ylabel("empirical CDF")
    axes[1].set_title(
        "b  Audit residuals are not pooled-exchangeable",
        loc="left", weight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].text(
        0.03, 0.55,
        f"audit coverage = {audit['covered'].mean():.5f}\n"
        "fixed quotas + de-duplication\npreclude a certificate",
        transform=axes[1].transAxes, fontsize=7.1,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white",
              "ec": COLORS["gray"], "alpha": 0.92},
    )
    fig.suptitle(
        "Calibration defines a frozen heuristic, not a realized coverage theorem",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure5_ResidualOrderStatistic")


def make_predicted_realized_figure(evidence: dict) -> list[Path]:
    audit = _routing_diagnostics(evidence["confirmatory"], "audit")
    accepted = audit["accepted"]
    if int(accepted.sum()) != 12:
        raise ValueError("accepted-route count differs from locked summary")
    selected_rows = np.flatnonzero(accepted)
    families = np.asarray(
        [row["family"] for row in audit["metadata"]], dtype=object)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), constrained_layout=True)
    for family in FAMILY_ORDER:
        mask = accepted & (families == family)
        if not mask.any():
            continue
        axes[0].scatter(
            audit["predicted_delta"][mask], audit["selected_delta"][mask],
            s=48, color=FAMILY_COLOR[family], edgecolor="white",
            linewidth=0.6, label=FAMILY_LABEL[family])
    lower = min(
        float(audit["predicted_delta"][accepted].min()),
        float(audit["selected_delta"][accepted].min()),
    ) - 0.005
    upper = max(
        float(audit["predicted_delta"][accepted].max()),
        float(audit["selected_delta"][accepted].max()),
    ) + 0.005
    axes[0].plot([lower, upper], [lower, upper],
                 color=COLORS["gray"], linestyle=":", lw=0.9)
    axes[0].axhline(0, color=COLORS["red"], lw=0.8)
    axes[0].axvline(-audit["quantile"], color=COLORS["red"],
                    linestyle="--", lw=0.9)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel(r"predicted $\widehat{\Delta}$")
    axes[0].set_ylabel(r"realized selected-minus-fallback $\Delta$")
    axes[0].set_title(
        "a  Accepted routes only (descriptive)", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=6.7)

    order = selected_rows[np.argsort(audit["selected_delta"][selected_rows])]
    realized = audit["selected_delta"][order]
    upper_bounds = audit["upper_bounds"][order]
    colors = [
        COLORS["red"] if value > 0 else COLORS["blue"]
        for value in realized
    ]
    y = np.arange(order.size)
    axes[1].hlines(
        y, upper_bounds, realized, color=COLORS["gray"], lw=1.0)
    axes[1].scatter(
        upper_bounds, y, marker="|", s=85, color=COLORS["orange"],
        label="predicted upper score")
    axes[1].scatter(
        realized, y, s=34, color=colors, edgecolor="white",
        linewidth=0.4, label="realized delta")
    axes[1].axvline(0, color=COLORS["ink"], lw=0.8)
    axes[1].set_yticks(
        y,
        [
            f"{FAMILY_LABEL[families[index]].split('-')[0]} "
            f"n={audit['metadata'][index]['n']}"
            for index in order
        ],
        fontsize=6.5,
    )
    axes[1].set_xlabel("AURC difference")
    axes[1].set_title(
        "b  Four realized harms despite negative gate scores",
        loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=6.7)
    fig.suptitle(
        "Predicted improvements do not uniformly transfer to accepted graphs",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure6_PredictedVsRealizedRoutes")


def make_anytime_family_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    checkpoints, curves = _confirmatory_curve_data(item)
    graph_rows = item["summary"]["splits"]["audit"]["per_graph"]
    families = np.asarray([row["family"] for row in graph_rows], dtype=object)
    styles = (
        ("global", "global arm", COLORS["gray"], "--"),
        ("family", "family fallback", COLORS["orange"], "-"),
        ("deployed", "gated deployment", COLORS["blue"], "-"),
        ("oracle", "checkpoint oracle", COLORS["green"], ":"),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.25, 5.45), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.23},
    )
    for ax, family in zip(axes.flat, FAMILY_ORDER):
        mask = families == family
        for key, label, color, linestyle in styles:
            mean = curves[key][mask].mean(axis=0)
            ax.plot(
                checkpoints, mean, marker="o", ms=3.0, lw=1.4,
                color=color, linestyle=linestyle, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(checkpoints, [str(value) for value in checkpoints])
        ax.grid(axis="y", color="#D5D9DD", lw=0.6)
        ax.set_title(
            f"{FAMILY_LABEL[family]} (n={int(mask.sum())})",
            loc="left", weight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean normalized regret")
    for ax in axes[1, :]:
        ax.set_xlabel("objective evaluations")
    axes[0, 0].legend(frameon=False, fontsize=6.7)
    fig.suptitle(
        "Anytime behavior differs across the four fixed graph families",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure7_AnytimeByFamily")


def make_family_effects_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    audit_summary = item["summary"]["splits"]["audit"]
    diagnostics = _routing_diagnostics(item, "audit")
    families = np.asarray(
        [row["family"] for row in diagnostics["metadata"]], dtype=object)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), constrained_layout=True)
    x = np.arange(len(FAMILY_ORDER))
    mean_effects = np.asarray([
        audit_summary["per_family"][family][
            "mean_delta_gated_minus_family_control"]
        for family in FAMILY_ORDER
    ])
    bars = axes[0].bar(
        x, mean_effects,
        color=[
            COLORS["blue"] if value < 0 else COLORS["red"]
            if value > 0 else COLORS["gray"]
            for value in mean_effects
        ],
        width=0.62,
    )
    axes[0].axhline(0, color=COLORS["ink"], lw=0.8)
    axes[0].set_xticks(x, ["ER", "RR", "BA", "WS"])
    axes[0].set_ylabel("mean gated-minus-fallback AURC")
    axes[0].set_title(
        "a  Family means (descriptive)", loc="left", weight="bold")
    axes[0].grid(axis="y", color="#D5D9DD", lw=0.6)
    for bar, value in zip(bars, mean_effects):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.0005 if value >= 0 else -0.0005),
            f"{value:+.4f}", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=7)

    accepted_counts = []
    harm_counts = []
    for family in FAMILY_ORDER:
        mask = families == family
        accepted_counts.append(int((diagnostics["accepted"] & mask).sum()))
        harm_counts.append(int((
            diagnostics["accepted"] & mask
            & (diagnostics["selected_delta"] > 0)
        ).sum()))
    width = 0.36
    axes[1].bar(
        x - width / 2, accepted_counts, width,
        color=COLORS["blue"], label="accepted")
    axes[1].bar(
        x + width / 2, harm_counts, width,
        color=COLORS["red"], label="harmful")
    axes[1].set_xticks(x, ["ER", "RR", "BA", "WS"])
    axes[1].set_ylabel("number of audit graphs")
    axes[1].set_title(
        "b  Acceptance and realized harm", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#D5D9DD", lw=0.6)
    fig.suptitle(
        "Pooled improvement masks concentrated benefit and harm",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure8_FamilyEffects")


def make_risk_sensitivity_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    risk = summarize_locked_audit(item["summary"])
    names = (
        "coverage failure",
        "joint harm",
        "conditional harm",
    )
    keys = (
        "coverage_failure",
        "joint_harm",
        "conditional_harm_given_acceptance",
    )
    empirical = np.asarray(
        [risk[key]["empirical_rate"] for key in keys], dtype=float)
    upper = np.asarray(
        [risk[key]["one_sided_clopper_pearson_upper"] for key in keys],
        dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), constrained_layout=True)
    x = np.arange(len(keys))
    width = 0.34
    axes[0].bar(
        x - width / 2, empirical, width,
        color=COLORS["blue"], label="empirical rate")
    axes[0].bar(
        x + width / 2, upper, width,
        color=COLORS["orange"], label="one-sided 95% upper bound")
    axes[0].axhline(
        0.10, color=COLORS["red"], linestyle="--", lw=1.0,
        label="risk threshold")
    axes[0].set_xticks(x, names, rotation=16, ha="right")
    axes[0].set_ylabel("event probability")
    axes[0].set_ylim(0, 0.68)
    axes[0].set_title(
        "a  Hypothetical iid-binomial sensitivity",
        loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=6.7)

    audit = item["summary"]["splits"]["audit"]
    covered = int(round(
        audit["empirical_one_sided_coverage"] * audit["n_graphs"]))
    counts = [covered, audit["n_graphs"] - covered]
    axes[1].bar(
        ["covered", "not covered"], counts,
        color=[COLORS["green"], COLORS["red"]], width=0.58)
    axes[1].axhline(
        0.90 * audit["n_graphs"], color=COLORS["red"],
        linestyle="--", lw=1.0, label="0.90 point requirement")
    for xpos, value in enumerate(counts):
        axes[1].text(
            xpos, value + 3, f"{value}/{audit['n_graphs']}",
            ha="center", fontsize=8, weight="bold")
    axes[1].set_ylim(0, audit["n_graphs"] * 1.05)
    axes[1].set_ylabel("audit graphs")
    axes[1].set_title(
        "b  Frozen point gate and model sensitivity",
        loc="left", weight="bold")
    axes[1].text(
        0.04, 0.55,
        "Binomial lower-tail probability\n"
        f"at p=0.90: {risk['coverage_lower_tail_probability_at_nominal']:.3f}\n"
        "diagnostic only; not a certificate",
        transform=axes[1].transAxes, fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white",
              "ec": COLORS["gray"]},
    )
    axes[1].legend(frameon=False, loc="upper right", fontsize=6.8)
    fig.suptitle(
        "Observed rates and confidence bounds answer different questions",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure9_RiskSensitivity")


def make_arm_transitions_figure(evidence: dict) -> list[Path]:
    diagnostics = _routing_diagnostics(evidence["confirmatory"], "audit")
    arms = diagnostics["arms"]
    active = [
        arm for arm in arms
        if np.any(diagnostics["selected_arms"] == arm)
        or np.any(diagnostics["deployed_arms"] == arm)
    ]
    active_index = {arm: index for index, arm in enumerate(active)}
    transitions = np.zeros((len(active), len(active)), dtype=int)
    for selected, deployed in zip(
            diagnostics["selected_arms"], diagnostics["deployed_arms"]):
        transitions[active_index[selected], active_index[deployed]] += 1

    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 3.75),
        gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.38},
    )
    image = axes[0].imshow(
        transitions, cmap="Blues", vmin=0,
        vmax=max(int(transitions.max()), 1), aspect="auto")
    axes[0].set_xticks(
        np.arange(len(active)), [ARM_LABEL[arm] for arm in active],
        rotation=25, ha="right")
    axes[0].set_yticks(
        np.arange(len(active)), [ARM_LABEL[arm] for arm in active])
    axes[0].set_xlabel("deployed arm")
    axes[0].set_ylabel("structurally selected arm")
    axes[0].set_title(
        "a  Selection-to-deployment transitions", loc="left", weight="bold")
    for row in range(transitions.shape[0]):
        for column in range(transitions.shape[1]):
            value = transitions[row, column]
            if value:
                axes[0].text(
                    column, row, str(value), ha="center", va="center",
                    color="white" if value > transitions.max() / 2
                    else COLORS["ink"],
                    fontsize=8, weight="bold")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="graphs")

    families = np.asarray(
        [row["family"] for row in diagnostics["metadata"]], dtype=object)
    accepted = np.asarray([
        int((diagnostics["accepted"] & (families == family)).sum())
        for family in FAMILY_ORDER
    ])
    total = np.asarray([
        int((families == family).sum()) for family in FAMILY_ORDER])
    abstained = total - accepted
    x = np.arange(len(FAMILY_ORDER))
    axes[1].bar(
        x, accepted, color=COLORS["blue"], label="accepted")
    axes[1].bar(
        x, abstained, bottom=accepted, color=COLORS["light"],
        label="fallback/abstained")
    axes[1].set_xticks(x, ["ER", "RR", "BA", "WS"])
    axes[1].set_ylabel("audit graphs")
    axes[1].set_title(
        "b  Abstention dominates deployment", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=6.8)
    axes[1].grid(axis="y", color="#D5D9DD", lw=0.6)
    fig.suptitle(
        "The gate converts most structural proposals back to family controls",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure10_ArmTransitions")


def make_split_diagnostics_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    matrices = {
        split: _split_trace_matrix(item, split)
        for split in ("development", "calibration", "audit")
    }
    mean = np.asarray(item["decision"]["selector"]["feature_mean"], dtype=float)
    scale = np.asarray(item["decision"]["selector"]["feature_scale"], dtype=float)
    standardized_means = np.stack([
        ((matrices[split]["features"] - mean) / scale).mean(axis=0)
        for split in ("development", "calibration", "audit")
    ])
    feature_labels = [
        name.replace("_scaled", "").replace("_", " ")
        for name in GRAPH_FEATURE_NAMES
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(7.25, 5.0),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.56},
    )
    limit = max(0.75, float(np.abs(standardized_means).max()))
    image = axes[0].imshow(
        standardized_means, cmap="coolwarm", vmin=-limit, vmax=limit,
        aspect="auto")
    axes[0].set_yticks(
        np.arange(3), ["development", "calibration", "audit"])
    axes[0].set_xticks(
        np.arange(len(feature_labels)), feature_labels,
        rotation=42, ha="right", fontsize=6.3)
    axes[0].set_title(
        "a  Split means in development-standardized feature units",
        loc="left", weight="bold")
    fig.colorbar(
        image, ax=axes[0], fraction=0.025, pad=0.02,
        label="standardized mean")

    categories = [
        (family, n) for family in FAMILY_ORDER for n in (12, 14)
    ]
    x = np.arange(len(categories))
    width = 0.25
    split_colors = (COLORS["gray"], COLORS["orange"], COLORS["blue"])
    for offset, split, color in zip(
            (-width, 0.0, width),
            ("development", "calibration", "audit"),
            split_colors):
        counts = [
            sum(
                row["family"] == family and row["n"] == n
                for row in matrices[split]["metadata"]
            )
            for family, n in categories
        ]
        axes[1].bar(
            x + offset, counts, width, color=color, label=split)
    axes[1].set_xticks(
        x, [f"{family.upper()}-{n}" for family, n in categories],
        rotation=25, ha="right")
    axes[1].set_ylabel("graphs")
    axes[1].set_title(
        "b  Fixed family-by-size quotas", loc="left", weight="bold")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(axis="y", color="#D5D9DD", lw=0.6)
    fig.suptitle(
        "Balanced strata aid diagnosis but do not imply pooled iid sampling",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure11_SplitDiagnostics")


def make_resource_ledger_figure(evidence: dict) -> list[Path]:
    summary = evidence["confirmatory"]["summary"]
    resources = summary["resource_accounting"]
    stages = (
        "angle labels",
        "development",
        "calibration",
        "audit",
    )
    queries = np.asarray([
        resources["angle_label_objective_queries"],
        resources["trace_objective_queries_by_split"]["development"],
        resources["trace_objective_queries_by_split"]["calibration"],
        resources["trace_objective_queries_by_split"]["audit"],
    ], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), constrained_layout=True)
    x = np.arange(len(stages))
    bars = axes[0].bar(
        x, queries,
        color=(COLORS["gray"], COLORS["orange"], COLORS["purple"], COLORS["blue"]))
    axes[0].set_xticks(x, stages, rotation=18, ha="right")
    axes[0].set_ylabel("exact objective calls")
    axes[0].set_title(
        "a  Offline and audit construction ledger",
        loc="left", weight="bold")
    axes[0].grid(axis="y", color="#D5D9DD", lw=0.6)
    for bar, value in zip(bars, queries):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value + 900,
            f"{value:,}", ha="center", fontsize=7)

    budget = int(resources["deployment_objective_query_ceiling"])
    arms = len(summary["arms"])
    comparison = [arms * budget, budget]
    bars = axes[1].bar(
        ["nine-arm audit\\nper graph", "one deployed arm\\nper graph"],
        comparison, color=[COLORS["red"], COLORS["green"]], width=0.58)
    axes[1].set_ylabel("objective-call ceiling")
    axes[1].set_title(
        "b  Ground-truth audit vs deployment",
        loc="left", weight="bold")
    axes[1].grid(axis="y", color="#D5D9DD", lw=0.6)
    for bar, value in zip(bars, comparison):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value + 6,
            str(value), ha="center", fontsize=8, weight="bold")
    axes[1].text(
        0.98, 0.82,
        "shots used: 0\nbackend: exact statevector\n"
        "no hardware-time conversion",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white",
              "ec": COLORS["gray"]},
    )
    fig.suptitle(
        "Equal online budgets do not erase offline experiment-construction cost",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure12_ResourceLedger")


def make_inference_threshold_figure(evidence: dict) -> list[Path]:
    item = evidence["confirmatory"]
    diagnostics = _routing_diagnostics(item, "audit")
    config = item["config"]
    groups = [
        f"{row['family']}:n{row['n']}" for row in diagnostics["metadata"]
    ]
    resamples = int(config["selector"]["bootstrap_resamples"])
    base_seed = int(config["seed"]) + 5000
    sensitivity_seeds = np.arange(40, dtype=int)
    intervals = np.asarray([
        _stratified_bootstrap_interval(
            diagnostics["deployed_delta"], groups,
            resamples=resamples, seed=base_seed + int(seed))
        for seed in sensitivity_seeds
    ])
    locked = np.asarray(
        item["summary"]["splits"]["audit"][
            "family_control_delta_bootstrap_95_interval"],
        dtype=float,
    )

    margins = np.linspace(-0.03, 0.03, 49)
    eligible = diagnostics["selected_arms"] != diagnostics["fallback_arms"]
    accepted_rate = []
    joint_harm_rate = []
    conditional_harm_rate = []
    for margin in margins:
        accepted = eligible & (diagnostics["upper_bounds"] < -margin)
        harms = accepted & (diagnostics["selected_delta"] > 0)
        accepted_rate.append(float(accepted.mean()))
        joint_harm_rate.append(float(harms.mean()))
        conditional_harm_rate.append(
            float(harms.sum() / accepted.sum()) if accepted.any() else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.45), constrained_layout=True)
    axes[0].fill_between(
        sensitivity_seeds, intervals[:, 0], intervals[:, 1],
        color=COLORS["light"], edgecolor=COLORS["blue"], linewidth=0.8,
        label="40 deterministic seed perturbations")
    axes[0].plot(
        sensitivity_seeds, intervals[:, 0], color=COLORS["blue"], lw=0.8)
    axes[0].plot(
        sensitivity_seeds, intervals[:, 1], color=COLORS["blue"], lw=0.8)
    axes[0].axhline(0, color=COLORS["red"], lw=0.9)
    axes[0].axhline(
        locked[0], color=COLORS["orange"], linestyle=":", lw=1.0)
    axes[0].axhline(
        locked[1], color=COLORS["orange"], linestyle=":", lw=1.0,
        label="locked percentile endpoints")
    axes[0].set_xlabel("exploratory resampling-seed index")
    axes[0].set_ylabel("95% percentile interval endpoint")
    axes[0].set_title(
        "a  Bootstrap seed sensitivity", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=6.7)

    axes[1].plot(
        margins, accepted_rate, color=COLORS["blue"], lw=1.6,
        label="accepted fraction")
    axes[1].plot(
        margins, joint_harm_rate, color=COLORS["green"], lw=1.6,
        label="joint-harm fraction")
    axes[1].plot(
        margins, conditional_harm_rate, color=COLORS["red"], lw=1.6,
        label="conditional harm")
    axes[1].axvline(
        0, color=COLORS["ink"], linestyle="--", lw=0.9,
        label="frozen margin")
    axes[1].set_xlabel("post-hoc additive acceptance margin")
    axes[1].set_ylabel("empirical fraction")
    axes[1].set_ylim(0, 0.72)
    axes[1].set_title(
        "b  Threshold trade-off (exploratory)",
        loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=6.6)
    axes[1].grid(axis="y", color="#D5D9DD", lw=0.6)
    fig.suptitle(
        "Sensitivity diagnostics do not replace the frozen confirmatory decision",
        x=0.01, ha="left", fontsize=11.5, weight="bold")
    return save_figure(fig, "Figure13_InferenceThresholdSensitivity")


def _tex_number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "-")


def make_tables_and_macros(evidence: dict) -> list[Path]:
    TABLES.mkdir(parents=True, exist_ok=True)
    item = evidence["confirmatory"]
    summary = item["summary"]
    audit = summary["splits"]["audit"]
    config = item["config"]
    manifest = item["manifest"]

    table_protocol = TABLES / "table_portfolio_protocol.tex"
    table_protocol.write_text(
        "\\begin{tabular}{@{}lrrrr@{}}\n"
        "\\toprule\n"
        "Split & Per family-size & Graphs & Exact queries & Role \\\\\n"
        "\\midrule\n"
        f"Angle labels & {config['per_family_size']['angle_train']} & 48 & "
        f"{summary['resource_accounting']['angle_label_objective_queries']:,} & legacy-arm training \\\\\n"
        f"Development & {config['per_family_size']['development']} & 48 & "
        f"{summary['resource_accounting']['trace_objective_queries_by_split']['development']:,} & routing and fallback \\\\\n"
        f"Calibration & {config['per_family_size']['calibration']} & 48 & "
        f"{summary['resource_accounting']['trace_objective_queries_by_split']['calibration']:,} & residual order statistic \\\\\n"
        f"Confirmatory audit & {config['per_family_size']['audit']} & 160 & "
        f"{summary['resource_accounting']['trace_objective_queries_by_split']['audit']:,} & one-shot decision \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    table_arms = TABLES / "table_portfolio_arms.tex"
    arm_rows = [
        ("Concentration", "coordinate-wise median label",
         "pattern search", "yes"),
        ("TQA $d_t=0.25$", "midpoint linear ramp",
         "pattern search", "no"),
        ("TQA $d_t=0.50$", "midpoint linear ramp",
         "pattern search", "no"),
        ("TQA $d_t=0.75$", "midpoint linear ramp",
         "pattern search", "no"),
        ("TQA $d_t=1.00$", "midpoint linear ramp",
         "pattern search", "no"),
        ("1-NN transfer", "nearest standardized training graph",
         "pattern search", "yes"),
        ("Random multistart", "four uniform random starts",
         "four pattern searches", "no"),
        ("SPSA", "concentration angle",
         "$a=0.18,c=0.12,\\alpha=0.602,\\gamma=0.101$", "yes"),
        ("Legacy GCTR", "GIN diagonal-Gaussian mean",
         "radius-two covariance-shaped pattern search", "yes"),
    ]
    table_arms.write_text(
        "\\begin{tabular}{@{}llllr@{}}\n"
        "\\toprule\n"
        "Arm & Initialization & Search rule & Offline labels & Calls \\\\\n"
        "\\midrule\n"
        + "\n".join(
            f"{arm} & {initialization} & {search} & {labels} & "
            f"{config['budget']} \\\\"
            for arm, initialization, search, labels in arm_rows
        )
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    table_results = TABLES / "table_confirmatory_results.tex"
    rows = []
    for family in FAMILY_ORDER:
        values = audit["per_family"][family]
        rows.append(
            f"{FAMILY_TEX_LABEL[family]} & {values['n_graphs']} & "
            f"{values['family_label_control_mean_aurc']:.5f} & "
            f"{values['gated_selector_mean_aurc']:.5f} & "
            f"{values['mean_delta_gated_minus_family_control']:+.5f} & "
            f"{values['accepted_count']} & {values['joint_harm_count']} \\\\"
        )
    table_results.write_text(
        "\\begin{tabular}{@{}lrrrrrr@{}}\n"
        "\\toprule\n"
        "Stratum & $N$ & Fallback & Gated & $\\Delta$ & Accepted & Harms \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\midrule\n"
        f"All & {audit['n_graphs']} & "
        f"{audit['family_label_control_mean_aurc']:.5f} & "
        f"{audit['gated_selector_mean_aurc']:.5f} & "
        f"{audit['mean_delta_gated_minus_family_label_control']:+.5f} & "
        f"{audit['accepted_count']} & {audit['joint_harm_count']} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    table_policy = TABLES / "table_policy_comparison.tex"
    family_mean = audit["family_label_control_mean_aurc"]
    policy_rows = [
        (
            "Global development arm",
            audit["baseline_mean_aurc"],
            audit["baseline_mean_aurc"] - family_mean,
            "deployable control",
        ),
        (
            "Family fallback",
            family_mean,
            0.0,
            "deployable control",
        ),
        (
            "Gated selector",
            audit["gated_selector_mean_aurc"],
            audit["mean_delta_gated_minus_family_label_control"],
            f"{audit['accepted_count']} accepted routes",
        ),
        (
            "Per-graph portfolio oracle",
            audit["oracle_mean_aurc"],
            audit["oracle_mean_aurc"] - family_mean,
            "retrospective only",
        ),
    ]
    table_policy.write_text(
        "\\begin{tabular}{@{}lrrl@{}}\n"
        "\\toprule\n"
        "Policy & Mean AURC & $\\Delta$ versus family & Interpretation \\\\\n"
        "\\midrule\n"
        + "\n".join(
            f"{label} & {mean:.5f} & {delta:+.5f} & {interpretation} \\\\"
            for label, mean, delta, interpretation in policy_rows
        )
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    gates = audit["go_no_go"]
    global_high = audit["delta_bootstrap_95_interval"][1]
    family_high = audit["family_control_delta_bootstrap_95_interval"][1]
    gate_rows = [
        ("Portfolio opportunity", "$\\geq 0.05$",
         f"{audit['portfolio_opportunity_fraction']:.3f}",
         gates["portfolio_opportunity_at_least_5pct"]),
        ("Global-baseline interval upper end", "$<0$",
         f"{global_high:.6f}", gates["gated_beats_global_baseline_95pct"]),
        ("Family-fallback interval upper end", "$<0$",
         f"{family_high:.8f}", gates["gated_beats_family_label_control_95pct"]),
        ("Accepted audit graphs", "$\\geq 10$",
         str(audit["accepted_count"]), gates["minimum_accepted_graphs_met"]),
        ("One-sided empirical coverage", "$\\geq 0.90$",
         f"{audit['empirical_one_sided_coverage']:.5f}",
         gates["empirical_coverage_at_least_nominal"]),
        ("Joint harm rate", "$\\leq 0.10$",
         f"{audit['joint_harm_rate']:.4f}",
         gates["joint_harm_rate_within_alpha"]),
    ]
    table_gate = TABLES / "table_confirmatory_gate.tex"
    table_gate.write_text(
        "\\begin{tabular}{@{}lrrc@{}}\n"
        "\\toprule\n"
        "Criterion & Required & Observed & Result \\\\\n"
        "\\midrule\n"
        + "\n".join(
            f"{label} & {required} & {observed} & "
            + ("pass" if passed else r"\textbf{fail}")
            + r" \\"
            for label, required, observed, passed in gate_rows
        )
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    resources = summary["resource_accounting"]
    table_resources = TABLES / "table_resource_accounting.tex"
    resource_rows = [
        (
            "Angle-label training",
            48,
            resources["angle_label_objective_queries"],
            "legacy-arm labels",
        ),
        (
            "Development traces",
            summary["splits"]["development"]["n_graphs"],
            resources["trace_objective_queries_by_split"]["development"],
            "nine-arm portfolio",
        ),
        (
            "Calibration traces",
            summary["splits"]["calibration"]["n_graphs"],
            resources["trace_objective_queries_by_split"]["calibration"],
            "nine-arm portfolio",
        ),
        (
            "Confirmatory traces",
            audit["n_graphs"],
            resources["trace_objective_queries_by_split"]["audit"],
            "nine-arm audit",
        ),
        (
            "Deployed policy",
            1,
            resources["deployment_objective_query_ceiling"],
            "per new graph",
        ),
    ]
    table_resources.write_text(
        "\\begin{tabular}{@{}lrrl@{}}\n"
        "\\toprule\n"
        "Stage & Graphs & Exact objective calls & Accounting role \\\\\n"
        "\\midrule\n"
        + "\n".join(
            f"{label} & {graphs} & {queries:,} & {role} \\\\"
            for label, graphs, queries, role in resource_rows
        )
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
    )

    macros = TABLES / "portfolio_numbers.tex"
    ci = audit["family_control_delta_bootstrap_95_interval"]
    dev = evidence["heterogeneous_development"]["summary"]["splits"]["audit"]
    macro_values = {
        "ConfirmN": str(audit["n_graphs"]),
        "ConfirmGlobalAURC": f"{audit['baseline_mean_aurc']:.5f}",
        "ConfirmGatedAURC": f"{audit['gated_selector_mean_aurc']:.5f}",
        "ConfirmFamilyAURC": f"{audit['family_label_control_mean_aurc']:.5f}",
        "ConfirmOracleAURC": f"{audit['oracle_mean_aurc']:.5f}",
        "ConfirmDelta": f"{audit['mean_delta_gated_minus_family_label_control']:.6f}",
        "ConfirmCILow": f"{ci[0]:.6f}",
        "ConfirmCIHigh": f"{ci[1]:.8f}",
        "ConfirmAccepted": str(audit["accepted_count"]),
        "ConfirmHarms": str(audit["joint_harm_count"]),
        "ConfirmJointHarmRate": f"{audit['joint_harm_rate']:.4f}",
        "ConfirmConditionalHarmRate": f"{audit['conditional_harm_rate']:.3f}",
        "ConfirmCoverage": f"{audit['empirical_one_sided_coverage']:.5f}",
        "ConfirmNominal": f"{1.0-config['selector']['alpha']:.2f}",
        "ConfirmAcceptedRate": f"{audit['accepted_rate']:.3f}",
        "ConfirmOpportunity": f"{audit['portfolio_opportunity_fraction']:.3f}",
        "ConfirmBAdelta": f"{audit['per_family']['ba']['mean_delta_gated_minus_family_control']:+.6f}",
        "ConfirmRRdelta": f"{audit['per_family']['rr']['mean_delta_gated_minus_family_control']:+.6f}",
        "ConfirmWSdelta": f"{audit['per_family']['ws']['mean_delta_gated_minus_family_control']:+.6f}",
        "DevDelta": f"{dev['mean_delta_gated_minus_family_label_control']:.6f}",
        "DevCILow": f"{dev['family_control_delta_bootstrap_95_interval'][0]:.6f}",
        "DevCIHigh": f"{dev['family_control_delta_bootstrap_95_interval'][1]:.6f}",
        "DevAccepted": str(dev["accepted_count"]),
        "DevHarms": str(dev["joint_harm_count"]),
        "DevCoverage": f"{dev['empirical_one_sided_coverage']:.4f}",
        "ProtocolHash": manifest["protocol_sha256"],
        "DecisionHash": manifest["decision_sha256"],
        "ImplementationHash": manifest["implementation_sha256"],
        "ConfigHash": manifest["config_sha256"],
    }
    macros.write_text("".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n"
        for name, value in macro_values.items()
    ))
    return [
        table_protocol,
        table_arms,
        table_results,
        table_policy,
        table_gate,
        table_resources,
        macros,
    ]


def write_artifact_manifest(
    paths: list[Path],
    evidence: dict,
    *,
    table_paths: list[Path],
) -> Path:
    manifest_path = MANUSCRIPT / "portfolio_artifacts_manifest.json"
    figure_paths = [
        path for path in paths if path.parent == FIGURES
    ]
    if len(figure_paths) != 2 * len(FIGURE_STEMS):
        raise ValueError("artifact set must contain PDF and PNG for 13 figures")
    if len(table_paths) != 6:
        raise ValueError("artifact set must contain exactly six study tables")
    body = {
        "schema_version": 2,
        "generator": "manuscript/generate_portfolio_artifacts.py",
        "asset_contract": {
            "figure_count": len(FIGURE_STEMS),
            "figure_formats": ["pdf", "png"],
            "figure_stems": list(FIGURE_STEMS),
            "table_count": len(table_paths),
            "table_files": [
                str(path.relative_to(ROOT)) for path in table_paths
            ],
            "all_quantitative_figures_derive_from_validated_locked_evidence": True,
            "exploratory_sensitivity_panels": [
                "Figure13_InferenceThresholdSensitivity",
            ],
        },
        "confirmatory_decision_sha256": evidence["confirmatory"]["manifest"][
            "decision_sha256"],
        "source_manifests": {
            name: item["manifest"]["decision_sha256"]
            for name, item in evidence.items()
        },
        "files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted(paths)
        },
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(body, indent=2) + "\n")
    return manifest_path


def main() -> int:
    configure_matplotlib()
    evidence = load_evidence()
    figure_makers = (
        make_protocol_figure,
        make_confirmatory_figure,
        make_progression_figure,
        make_arm_opportunity_figure,
        make_residual_order_figure,
        make_predicted_realized_figure,
        make_anytime_family_figure,
        make_family_effects_figure,
        make_risk_sensitivity_figure,
        make_arm_transitions_figure,
        make_split_diagnostics_figure,
        make_resource_ledger_figure,
        make_inference_threshold_figure,
    )
    if len(figure_makers) != len(FIGURE_STEMS):
        raise ValueError("generator must define exactly 13 figure makers")
    outputs = []
    for maker, expected_stem in zip(figure_makers, FIGURE_STEMS):
        generated = maker(evidence)
        if {path.stem for path in generated} != {expected_stem}:
            raise ValueError(
                f"{maker.__name__} did not generate {expected_stem}")
        outputs.extend(generated)
    table_outputs = make_tables_and_macros(evidence)
    table_paths = [
        path for path in table_outputs
        if path.name.startswith("table_")
    ]
    if len(table_paths) != 6:
        raise ValueError("generator must expose exactly six manuscript tables")
    outputs.extend(table_outputs)
    manifest = write_artifact_manifest(
        outputs, evidence, table_paths=table_paths)
    print(f"[portfolio-paper] generated {len(outputs)} artifacts")
    print("[portfolio-paper] figures " + ", ".join(FIGURE_STEMS))
    print("[portfolio-paper] tables " + ", ".join(
        path.name for path in table_paths))
    print(f"[portfolio-paper] manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
