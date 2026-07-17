"""Runnable target-free optimizer-portfolio experiment and analyzer."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

from .portfolio import (
    ARM_ORDER,
    ConformalAbstainer,
    PortfolioContext,
    RidgeUtilitySelector,
    build_portfolio_context,
    generate_angle_label,
    graph_summary_features,
    instance_key,
    run_portfolio_arm,
)
from .protocol import (
    instantiate_protocol_split,
    load_and_verify_protocol,
    sha256_json,
    write_frozen_protocol,
)
from .provenance import compatible_portfolio_fingerprints


RESULT_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portfolio_implementation_fingerprint() -> str:
    """Hash every source module that can affect portfolio-v1 evidence."""
    package_dir = Path(__file__).resolve().parent
    names = {
        "fixed_budget.py", "portfolio.py", "portfolio_experiment.py",
        "protocol.py", "graphs.py", "qaoa.py", "optimization.py",
        "model.py",
    }
    digest = hashlib.sha256()
    for path in sorted((package_dir / name for name in names),
                       key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _environment() -> dict:
    import networkx
    import scipy
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "torch": torch.__version__,
    }


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    required = {
        "schema_version", "experiment_name", "families", "sizes",
        "per_family_size", "seed_start", "budget", "checkpoints",
        "p_depth", "angle_label_starts", "angle_label_budget",
        "model", "selector", "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"portfolio config missing keys: {missing}")
    return config


def _trace_record(split, instance, arm, trace, features, optimizer_seed,
                  generator_parameters=None):
    ratios = trace.ratios(instance.maxcut)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "split": split,
        "graph_id": instance_key(instance),
        "family": instance.family,
        "n": int(instance.n),
        "graph_seed": int(instance.seed),
        "generator_parameters": generator_parameters,
        "maxcut": int(instance.maxcut),
        "arm": arm,
        "checkpoints": list(trace.checkpoints),
        "best_values": list(trace.best_values),
        "expectation_ratios": list(ratios),
        "best_params": [list(values) for values in trace.best_params],
        "aurc": trace.aurc(instance.maxcut),
        "evaluations_used": trace.evaluations_used,
        "objective_queries": trace.objective_queries,
        "shots_used": trace.shots_used,
        "backend": trace.backend,
        "optimizer_seed": int(optimizer_seed),
        "graph_features": [float(value) for value in features],
    }


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n"
                            for record in records))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _split_matrices(records: list[dict], split: str, arms=ARM_ORDER):
    selected = [record for record in records if record["split"] == split]
    graph_ids = sorted({record["graph_id"] for record in selected})
    by_pair = {(record["graph_id"], record["arm"]): record
               for record in selected}
    features = []
    utilities = []
    metadata = []
    for graph_id in graph_ids:
        rows = [by_pair[(graph_id, arm)] for arm in arms]
        features.append(rows[0]["graph_features"])
        utilities.append([row["aurc"] for row in rows])
        metadata.append({key: rows[0][key] for key in
                         ("graph_id", "family", "n", "graph_seed", "maxcut",
                          "generator_parameters")})
    return (graph_ids, np.asarray(features, dtype=float),
            np.asarray(utilities, dtype=float), metadata)


def _selector_dict(selector: RidgeUtilitySelector) -> dict:
    return {
        "arms": list(selector.arms),
        "feature_mean": selector.feature_mean.tolist(),
        "feature_scale": selector.feature_scale.tolist(),
        "coefficients": selector.coefficients.tolist(),
        "ridge": selector.ridge,
    }


def _selector_from_dict(value: dict) -> RidgeUtilitySelector:
    return RidgeUtilitySelector(
        arms=tuple(value["arms"]),
        feature_mean=np.asarray(value["feature_mean"], dtype=float),
        feature_scale=np.asarray(value["feature_scale"], dtype=float),
        coefficients=np.asarray(value["coefficients"], dtype=float),
        ridge=float(value["ridge"]),
    )


def freeze_decision(
    records: list[dict],
    config: dict,
    protocol: dict,
    implementation_sha256: str,
    legacy_model_sha256: str,
) -> dict:
    _, development_features, development_utilities, development_metadata = _split_matrices(
        records, "development")
    _, calibration_features, calibration_utilities, calibration_metadata = _split_matrices(
        records, "calibration")
    mean_by_arm = development_utilities.mean(axis=0)
    baseline_index = int(np.argmin(mean_by_arm))
    baseline = ARM_ORDER[baseline_index]
    selector = RidgeUtilitySelector.fit(
        development_features,
        development_utilities,
        ARM_ORDER,
        ridge=float(config["selector"]["ridge"]),
    )
    family_control = {}
    for family in sorted({row["family"] for row in development_metadata}):
        mask = np.asarray([row["family"] == family
                           for row in development_metadata], dtype=bool)
        family_control[family] = ARM_ORDER[int(np.argmin(
            development_utilities[mask].mean(axis=0)))]
    selected, calibration_predictions = selector.choose(calibration_features)
    calibration_baselines = [
        family_control[row["family"]] for row in calibration_metadata]
    abstainer = ConformalAbstainer.calibrate(
        baseline_arms=calibration_baselines,
        selected_arms=selected,
        predicted_utilities=calibration_predictions,
        actual_utilities=calibration_utilities,
        arms=ARM_ORDER,
        alpha=float(config["selector"]["alpha"]),
        margin=float(config["selector"].get("margin", 0.0)),
    )
    non_audit = [record for record in records if record["split"] != "audit"]
    decision = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_sha256": implementation_sha256,
        "legacy_model_sha256": legacy_model_sha256,
        "non_audit_trace_sha256": sha256_json(non_audit),
        "development_mean_aurc": {
            arm: float(mean_by_arm[i]) for i, arm in enumerate(ARM_ORDER)},
        "baseline_arm": baseline,
        "family_control_arm_by_family": family_control,
        "deployment_baseline_policy": "development_selected_per_family_arm",
        "selector": _selector_dict(selector),
        "abstainer": {
            "alpha": abstainer.alpha,
            "margin": abstainer.margin,
            "residual_quantile": abstainer.residual_quantile,
            "baseline_arm": abstainer.baseline_arm,
        },
    }
    decision["decision_sha256"] = sha256_json(decision)
    return decision


def _bootstrap_interval(
    values: np.ndarray,
    *,
    groups: Iterable[str] | None = None,
    resamples: int,
    seed: int,
):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    labels = (np.asarray(list(groups), dtype=object)
              if groups is not None else np.zeros(data.size, dtype=int))
    if labels.shape != (data.size,):
        raise ValueError("bootstrap groups must match values")
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for index in range(int(resamples)):
        sample = np.concatenate([
            rng.choice(stratum, size=stratum.size, replace=True)
            for stratum in strata
        ])
        means[index] = data[sample].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def analyze_records(
    records: list[dict],
    decision: dict,
    config: dict,
) -> dict:
    selector = _selector_from_dict(decision["selector"])
    abstainer = ConformalAbstainer(**decision["abstainer"])
    arm_index = {arm: index for index, arm in enumerate(ARM_ORDER)}
    baseline_index = arm_index[decision["baseline_arm"]]

    split_summaries = {}
    for split in ("development", "calibration", "audit"):
        graph_ids, features, utilities, metadata = _split_matrices(records, split)
        split_summaries[split] = {
            "n_graphs": len(graph_ids),
            "mean_aurc_by_arm": {
                arm: float(utilities[:, i].mean())
                for i, arm in enumerate(ARM_ORDER)},
        }
        if split != "audit":
            continue
        selected, predictions = selector.choose(features)
        row = np.arange(len(graph_ids))
        selected_indices = np.asarray([arm_index[name] for name in selected])
        baseline_utilities = utilities[:, baseline_index]
        selected_utilities = utilities[row, selected_indices]
        oracle_utilities = utilities.min(axis=1)
        family_control_arms = np.asarray([
            decision["family_control_arm_by_family"].get(
                item["family"], decision["baseline_arm"])
            for item in metadata
        ], dtype=object)
        family_control_indices = np.asarray(
            [arm_index[name] for name in family_control_arms])
        family_control_utilities = utilities[row, family_control_indices]
        deployed, accepted, upper = abstainer.deploy(
            selected,
            predictions,
            ARM_ORDER,
            baseline_arms=family_control_arms,
        )
        deployed_indices = np.asarray([arm_index[name] for name in deployed])
        deployed_utilities = utilities[row, deployed_indices]
        deployed_delta = deployed_utilities - baseline_utilities
        family_delta = deployed_utilities - family_control_utilities
        accepted_harm = accepted & (family_delta > 0.0)
        selected_delta = selected_utilities - family_control_utilities
        empirical_coverage = selected_delta <= upper + 1e-15
        bootstrap_groups = [f'{item["family"]}:n{item["n"]}'
                            for item in metadata]
        interval = _bootstrap_interval(
            deployed_delta,
            groups=bootstrap_groups,
            resamples=int(config["selector"].get("bootstrap_resamples", 4000)),
            seed=int(config["seed"]) + 991,
        )
        family_interval = _bootstrap_interval(
            family_delta,
            groups=bootstrap_groups,
            resamples=int(config["selector"].get("bootstrap_resamples", 4000)),
            seed=int(config["seed"]) + 992,
        )
        denominator = float(baseline_utilities.mean() - oracle_utilities.mean())
        captured = (float(baseline_utilities.mean() - deployed_utilities.mean())
                    / denominator if denominator > 1e-12 else 0.0)
        stratum_summary = {}
        for family in sorted({item["family"] for item in metadata}):
            mask = np.asarray([item["family"] == family
                               for item in metadata], dtype=bool)
            stratum_summary[family] = {
                "n_graphs": int(mask.sum()),
                "baseline_mean_aurc": float(baseline_utilities[mask].mean()),
                "gated_selector_mean_aurc": float(
                    deployed_utilities[mask].mean()),
                "family_label_control_mean_aurc": float(
                    family_control_utilities[mask].mean()),
                "family_control_arm": str(family_control_arms[mask][0]),
                "oracle_mean_aurc": float(oracle_utilities[mask].mean()),
                "mean_delta_gated_minus_baseline": float(
                    deployed_delta[mask].mean()),
                "mean_delta_gated_minus_family_control": float(
                    family_delta[mask].mean()),
                "accepted_count": int(accepted[mask].sum()),
                "joint_harm_count": int(accepted_harm[mask].sum()),
            }
        opportunity_fraction = (
            denominator / float(baseline_utilities.mean())
            if baseline_utilities.mean() > 1e-12 else 0.0)
        go_no_go = {
            "portfolio_opportunity_at_least_5pct": bool(
                opportunity_fraction >= 0.05),
            "gated_beats_global_baseline_95pct": bool(interval[1] < 0.0),
            "gated_beats_family_label_control_mean": bool(
                family_delta.mean() < 0.0),
            "gated_beats_family_label_control_95pct": bool(
                family_interval[1] < 0.0),
            "minimum_accepted_graphs_met": bool(
                accepted.sum() >= int(config["selector"].get(
                    "minimum_accepted", 10))),
            "empirical_coverage_at_least_nominal": bool(
                empirical_coverage.mean() >= 1.0 - abstainer.alpha),
            "joint_harm_rate_within_alpha": bool(
                accepted_harm.mean() <= abstainer.alpha),
        }
        required_gates = (
            "portfolio_opportunity_at_least_5pct",
            "gated_beats_global_baseline_95pct",
            "gated_beats_family_label_control_95pct",
            "minimum_accepted_graphs_met",
            "empirical_coverage_at_least_nominal",
            "joint_harm_rate_within_alpha",
        )
        go_no_go["confirmatory_followup_ready"] = bool(
            all(go_no_go[name] for name in required_gates))
        # A successful development pilot is still not a journal-readiness
        # determination. Only a separately labelled, preregistered
        # confirmatory configuration can pass this mechanical field.
        go_no_go["nmi_claim_ready"] = bool(
            go_no_go["confirmatory_followup_ready"]
            and config.get("study_stage") == "confirmatory")
        split_summaries[split].update({
            "baseline_arm": decision["baseline_arm"],
            "baseline_mean_aurc": float(baseline_utilities.mean()),
            "ungated_selector_mean_aurc": float(selected_utilities.mean()),
            "gated_selector_mean_aurc": float(deployed_utilities.mean()),
            "family_label_control_mean_aurc": float(
                family_control_utilities.mean()),
            "mean_delta_gated_minus_family_label_control": float(
                family_delta.mean()),
            "family_control_delta_bootstrap_95_interval": list(family_interval),
            "oracle_mean_aurc": float(oracle_utilities.mean()),
            "mean_delta_gated_minus_baseline": float(deployed_delta.mean()),
            "delta_bootstrap_95_interval": list(interval),
            "median_delta_gated_minus_baseline": float(
                np.median(deployed_delta)),
            "accepted_count": int(accepted.sum()),
            "accepted_rate": float(accepted.mean()),
            "joint_harm_count": int(accepted_harm.sum()),
            "joint_harm_rate": float(accepted_harm.mean()),
            "conditional_harm_rate": (
                float(accepted_harm.sum() / accepted.sum())
                if accepted.sum() else 0.0),
            "empirical_one_sided_coverage": float(empirical_coverage.mean()),
            "oracle_gap_captured_fraction": captured,
            "selected_arm_counts": {
                arm: int(np.sum(selected == arm)) for arm in ARM_ORDER},
            "deployed_arm_counts": {
                arm: int(np.sum(deployed == arm)) for arm in ARM_ORDER},
            "family_control_arm_by_family":
                decision["family_control_arm_by_family"],
            "deployment_baseline_policy":
                decision["deployment_baseline_policy"],
            "mean_conformal_upper_bound": float(upper.mean()),
            "portfolio_opportunity_fraction": opportunity_fraction,
            "per_family": stratum_summary,
            "go_no_go": go_no_go,
            "per_graph": [
                {
                    **metadata[i],
                    "selected_arm": str(selected[i]),
                    "deployed_arm": str(deployed[i]),
                    "family_control_arm": str(family_control_arms[i]),
                    "accepted": bool(accepted[i]),
                    "upper_bound": float(upper[i]),
                    "baseline_aurc": float(baseline_utilities[i]),
                    "family_control_aurc": float(
                        family_control_utilities[i]),
                    "deployed_aurc": float(deployed_utilities[i]),
                    "delta": float(deployed_delta[i]),
                    "delta_vs_family_control": float(family_delta[i]),
                }
                for i in range(len(graph_ids))
            ],
        })

    query_totals = {}
    for split in ("development", "calibration", "audit"):
        rows = [record for record in records if record["split"] == split]
        query_totals[split] = int(sum(row["objective_queries"] for row in rows))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "target_free": True,
        "backend": "exact_statevector",
        "environment": _environment(),
        "primary_metric": "equal_weight_checkpointed_normalized_regret_aurc",
        "checkpoints": config["checkpoints"],
        "fixed_budget": config["budget"],
        "arms": list(ARM_ORDER),
        "decision_sha256": decision["decision_sha256"],
        "conformal": decision["abstainer"],
        "splits": split_summaries,
        "resource_accounting": {
            "trace_objective_queries_by_split": query_totals,
            "shots_used": 0,
            "deployment_objective_query_ceiling": int(config["budget"]),
            "audit_maxcut_used_for_analysis_only": True,
        },
    }


def run_experiment(config: dict, output_dir: Path, protocol: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "traces.jsonl").exists():
        raise RuntimeError(
            "audit traces already exist; use a new output directory rather "
            "than unlocking the same audit twice")
    implementation_sha256 = portfolio_implementation_fingerprint()
    split_instances = {
        split: instantiate_protocol_split(protocol, split)
        for split in ("angle_train", "development", "calibration", "audit")}
    protocol_metadata = {
        (split, record["family"], record["n"], record["seed"]):
            record.get("generator_parameters")
        for split, split_records in protocol["splits"].items()
        for record in split_records
    }

    label_records = []
    target_angles = []
    label_queries = 0
    for index, instance in enumerate(split_instances["angle_train"]):
        params, value, queries = generate_angle_label(
            instance,
            p_depth=int(config["p_depth"]),
            n_starts=int(config["angle_label_starts"]),
            budget_per_start=int(config["angle_label_budget"]),
            seed=int(config["seed"]) + index,
        )
        target_angles.append(params)
        label_queries += queries
        label_records.append({
            "graph_id": instance_key(instance),
            "family": instance.family,
            "n": instance.n,
            "graph_seed": instance.seed,
            "params": params.tolist(),
            "value": float(value),
            "objective_queries": int(queries),
        })
    label_path = output_dir / "angle_labels.json"
    label_path.write_text(json.dumps({
        "schema_version": RESULT_SCHEMA_VERSION,
        "objective_queries": label_queries,
        "labels": label_records,
    }, indent=2) + "\n")

    evaluation_instances = [
        instance
        for split in ("development", "calibration", "audit")
        for instance in split_instances[split]
    ]
    model_config = config["model"]
    context: PortfolioContext = build_portfolio_context(
        split_instances["angle_train"],
        np.asarray(target_angles),
        evaluation_instances,
        p_depth=int(config["p_depth"]),
        spectral_k=int(model_config["spectral_k"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        epochs=int(model_config["epochs"]),
        learning_rate=float(model_config["learning_rate"]),
        seed=int(config["seed"]),
    )
    model_path = output_dir / "legacy_model_state.pt"
    torch.save(context.legacy_model.state_dict(), model_path)
    legacy_model_sha256 = _sha256_file(model_path)

    records: list[dict] = []

    def evaluate_split(split: str) -> None:
        for graph_index, instance in enumerate(split_instances[split]):
            features = graph_summary_features(instance.graph)
            for arm_index, arm in enumerate(ARM_ORDER):
                optimizer_seed = (int(config["seed"]) + instance.seed * 17
                                  + graph_index * 101 + arm_index * 1009)
                trace = run_portfolio_arm(
                    arm,
                    instance,
                    context,
                    budget=int(config["budget"]),
                    checkpoints=config["checkpoints"],
                    seed=optimizer_seed,
                )
                records.append(_trace_record(
                    split, instance, arm, trace, features, optimizer_seed,
                    protocol_metadata[(split, instance.family, instance.n,
                                       instance.seed)]))

    # Freeze the decision rule before any audit outcome is generated.
    evaluate_split("development")
    evaluate_split("calibration")
    non_audit_path = output_dir / "non_audit_traces.jsonl"
    _write_jsonl(non_audit_path, records)
    decision = freeze_decision(
        records, config, protocol, implementation_sha256,
        legacy_model_sha256)
    decision_path = output_dir / "frozen_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")

    evaluate_split("audit")
    traces_path = output_dir / "traces.jsonl"
    _write_jsonl(traces_path, records)
    summary = analyze_records(records, decision, config)
    summary["protocol_sha256"] = protocol["protocol_sha256"]
    summary["implementation_sha256"] = implementation_sha256
    summary["resource_accounting"]["angle_label_objective_queries"] = label_queries
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_sha256": implementation_sha256,
        "config_sha256": sha256_json(config),
        "decision_sha256": decision["decision_sha256"],
        "environment": _environment(),
        "files": {
            path.name: _sha256_file(path)
            for path in (label_path, model_path, non_audit_path, decision_path,
                         traces_path, summary_path)
        },
    }
    manifest_path = output_dir / "portfolio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def validate_portfolio_manifest(output_dir: Path, config: dict | None = None):
    """Verify the prospective decision, evidence files and implementation."""
    manifest_path = output_dir / "portfolio_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors = []
    implementation_sha256 = portfolio_implementation_fingerprint()
    if manifest.get("implementation_sha256") not in \
            compatible_portfolio_fingerprints(implementation_sha256):
        errors.append("portfolio implementation hash differs")
    if config is not None and manifest.get("config_sha256") != sha256_json(config):
        errors.append("portfolio config hash differs")
    for name, expected in manifest.get("files", {}).items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing evidence file: {name}")
        elif _sha256_file(path) != expected:
            errors.append(f"hash mismatch: {name}")
    protocol = json.loads((output_dir / "frozen_protocol.json").read_text())
    decision = json.loads((output_dir / "frozen_decision.json").read_text())
    protocol_body = {key: value for key, value in protocol.items()
                     if key != "protocol_sha256"}
    if sha256_json(protocol_body) != protocol.get("protocol_sha256"):
        errors.append("frozen protocol self-hash differs")
    decision_body = {key: value for key, value in decision.items()
                     if key != "decision_sha256"}
    if sha256_json(decision_body) != decision.get("decision_sha256"):
        errors.append("frozen decision self-hash differs")
    if protocol.get("protocol_sha256") != manifest.get("protocol_sha256"):
        errors.append("protocol hash differs from manifest")
    if decision.get("decision_sha256") != manifest.get("decision_sha256"):
        errors.append("decision hash differs from manifest")
    if decision.get("legacy_model_sha256") != manifest.get("files", {}).get(
            "legacy_model_state.pt"):
        errors.append("legacy model hash differs from frozen decision")
    if errors:
        raise RuntimeError("portfolio validation failed:\n- " + "\n- ".join(errors))
    return manifest


def analyze_existing(config: dict, output_dir: Path) -> dict:
    records = _read_jsonl(output_dir / "traces.jsonl")
    decision = json.loads((output_dir / "frozen_decision.json").read_text())
    recorded_hash = decision["decision_sha256"]
    body = {key: value for key, value in decision.items()
            if key != "decision_sha256"}
    if recorded_hash != sha256_json(body):
        raise RuntimeError("frozen decision hash mismatch")
    summary = analyze_records(records, decision, config)
    return summary


def run(argv=None) -> dict | None:
    parser = argparse.ArgumentParser(
        prog="gctr-portfolio",
        description="Freeze and run the target-free GCTR optimizer portfolio.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "frozen_protocol.json"

    if args.freeze_only:
        if protocol_path.exists():
            protocol = load_and_verify_protocol(config, protocol_path)
            print("[portfolio] protocol already frozen",
                  protocol["protocol_sha256"])
        else:
            protocol = write_frozen_protocol(config, protocol_path)
            print("[portfolio] froze protocol", protocol["protocol_sha256"])
        return protocol
    if args.validate_only:
        manifest = validate_portfolio_manifest(output_dir, config)
        print("[portfolio] valid", manifest["decision_sha256"])
        return manifest
    if args.analyze_only:
        validate_portfolio_manifest(output_dir, config)
        summary = analyze_existing(config, output_dir)
        print(json.dumps(summary["splits"]["audit"], indent=2))
        return summary
    if not protocol_path.exists():
        if args.run_only:
            raise RuntimeError("--run-only requires frozen_protocol.json")
        protocol = write_frozen_protocol(config, protocol_path)
        print("[portfolio] froze protocol", protocol["protocol_sha256"])
    else:
        protocol = load_and_verify_protocol(config, protocol_path)
        print("[portfolio] verified protocol", protocol["protocol_sha256"])
    summary = run_experiment(config, output_dir, protocol)
    audit = summary["splits"]["audit"]
    print("[portfolio] baseline", audit["baseline_arm"],
          "AURC", f'{audit["baseline_mean_aurc"]:.6f}')
    print("[portfolio] gated selector AURC",
          f'{audit["gated_selector_mean_aurc"]:.6f}',
          "delta", f'{audit["mean_delta_gated_minus_baseline"]:.6f}',
          "accepted", f'{audit["accepted_count"]}/{audit["n_graphs"]}')
    return summary


def main(argv=None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
