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
    ax.text(0, 0.985, "A prospective, target-free decision path",
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
            f"{'pass' if passed else '\\textbf{fail}'} \\\\"
            for label, required, observed, passed in gate_rows
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
    return [table_protocol, table_results, table_gate, macros]


def write_artifact_manifest(paths: list[Path], evidence: dict) -> Path:
    manifest_path = MANUSCRIPT / "portfolio_artifacts_manifest.json"
    body = {
        "schema_version": 1,
        "generator": "manuscript/generate_portfolio_artifacts.py",
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
    outputs = []
    outputs.extend(make_protocol_figure(evidence))
    outputs.extend(make_confirmatory_figure(evidence))
    outputs.extend(make_progression_figure(evidence))
    outputs.extend(make_tables_and_macros(evidence))
    manifest = write_artifact_manifest(outputs, evidence)
    print(f"[portfolio-paper] generated {len(outputs)} artifacts")
    print(f"[portfolio-paper] manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
