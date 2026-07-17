import json

import networkx as nx
import numpy as np
import pytest

from specops_gctr.portfolio import (
    ARM_ORDER,
    ConformalAbstainer,
    RidgeUtilitySelector,
    TQA_DT_BY_ARM,
    finite_sample_upper_quantile,
    graph_summary_features,
)
from specops_gctr.portfolio_experiment import (
    run_experiment,
    validate_portfolio_manifest,
)
from specops_gctr.protocol import (
    build_protocol,
    load_and_verify_protocol,
    write_frozen_protocol,
)
from specops_gctr.provenance import compatible_portfolio_fingerprints


def _tiny_config():
    return {
        "schema_version": 1,
        "study_stage": "development",
        "experiment_name": "portfolio-test",
        "families": ["er", "ws"],
        "sizes": [8],
        "per_family_size": {
            "angle_train": 1,
            "development": 1,
            "calibration": 1,
            "audit": 1,
        },
        "seed_start": {
            "angle_train": 51000,
            "development": 52000,
            "calibration": 53000,
            "audit": 54000,
        },
        "p_depth": 2,
        "angle_label_starts": 1,
        "angle_label_budget": 6,
        "budget": 4,
        "checkpoints": [1, 2, 4],
        "model": {
            "spectral_k": 3,
            "hidden_dim": 8,
            "num_layers": 1,
            "epochs": 5,
            "learning_rate": 0.005,
        },
        "selector": {
            "ridge": 1.0,
            "alpha": 0.25,
            "margin": 0.0,
            "minimum_accepted": 1,
            "bootstrap_resamples": 50,
        },
        "seed": 20260716,
    }


def test_graph_summary_is_invariant_to_node_relabelling():
    graph = nx.barabasi_albert_graph(10, 2, seed=11)
    mapping = {node: (7 * node + 3) % 10 for node in graph.nodes()}
    relabelled = nx.relabel_nodes(graph, mapping)
    np.testing.assert_allclose(
        graph_summary_features(graph), graph_summary_features(relabelled),
        atol=1e-12, rtol=1e-12)


def test_frozen_protocol_detects_config_drift(tmp_path):
    config = _tiny_config()
    path = tmp_path / "protocol.json"
    frozen = write_frozen_protocol(config, path)
    assert frozen == load_and_verify_protocol(config, path)
    changed = json.loads(json.dumps(config))
    changed["budget"] = 5
    changed["checkpoints"][-1] = 5
    with pytest.raises(RuntimeError, match="config differs"):
        load_and_verify_protocol(changed, path)


def test_continuous_generator_parameters_are_frozen_and_replayable(tmp_path):
    config = _tiny_config()
    config["generator_parameter_ranges"] = {
        "er": {"p": [0.2, 0.8]},
        "ws": {"degrees": [2, 4, 6], "rewiring": [0.05, 0.9]},
    }
    protocol = build_protocol(config)
    values = [record["generator_parameters"]
              for records in protocol["splits"].values()
              for record in records]
    assert all(value is not None for value in values)
    path = tmp_path / "heterogeneous_protocol.json"
    path.write_text(json.dumps(protocol))
    assert load_and_verify_protocol(config, path) == protocol


def test_ridge_selector_and_one_sided_abstention():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    utilities = np.array([
        [0.20, 0.30],
        [0.20, 0.22],
        [0.20, 0.12],
        [0.20, 0.05],
    ])
    selector = RidgeUtilitySelector.fit(
        features, utilities, ("baseline", "adaptive"), ridge=0.1)
    selected, predicted = selector.choose(features)
    abstainer = ConformalAbstainer.calibrate(
        baseline_arm="baseline",
        selected_arms=selected,
        predicted_utilities=predicted,
        actual_utilities=utilities,
        arms=("baseline", "adaptive"),
        alpha=0.25,
    )
    deployed, accepted, upper = abstainer.deploy(
        selected, predicted, ("baseline", "adaptive"))
    assert deployed.shape == accepted.shape == upper.shape == (4,)
    assert np.all((deployed == "baseline") | accepted)
    assert finite_sample_upper_quantile(np.arange(4.0), 0.25) == 3.0


def test_finite_sample_quantile_returns_infinity_when_rank_exceeds_sample():
    assert finite_sample_upper_quantile(
        np.asarray([0.1, 0.2]), alpha=0.1) == float("inf")
    unknown_core = "0" * 64
    assert compatible_portfolio_fingerprints(unknown_core) == \
        frozenset((unknown_core,))


def test_tqa_variants_are_explicit_and_family_baselines_are_row_specific():
    assert tuple(TQA_DT_BY_ARM) == tuple(
        arm for arm in ARM_ORDER if arm.startswith("tqa_dt_"))
    assert len(set(TQA_DT_BY_ARM.values())) == 4
    arms = ("family_a", "family_b", "candidate")
    predicted = np.array([
        [0.20, 0.40, 0.10],
        [0.40, 0.20, 0.10],
        [0.20, 0.40, 0.15],
        [0.40, 0.20, 0.15],
    ])
    actual = predicted.copy()
    baselines = ("family_a", "family_b", "family_a", "family_b")
    selected = ("candidate",) * 4
    abstainer = ConformalAbstainer.calibrate(
        baseline_arms=baselines,
        selected_arms=selected,
        predicted_utilities=predicted,
        actual_utilities=actual,
        arms=arms,
        alpha=0.25,
    )
    deployed, accepted, upper = abstainer.deploy(
        selected, predicted, arms, baseline_arms=baselines)
    assert np.all(deployed == "candidate")
    assert np.all(accepted)
    np.testing.assert_allclose(upper, [-0.10, -0.10, -0.05, -0.05])


def test_tiny_end_to_end_experiment_freezes_then_audits(tmp_path):
    config = _tiny_config()
    protocol = write_frozen_protocol(config, tmp_path / "frozen_protocol.json")
    summary = run_experiment(config, tmp_path, protocol)
    assert summary["target_free"] is True
    assert summary["splits"]["audit"]["n_graphs"] == 2
    assert "family_label_control_mean_aurc" in summary["splits"]["audit"]
    assert summary["resource_accounting"]["shots_used"] == 0
    assert (tmp_path / "frozen_decision.json").is_file()
    assert (tmp_path / "traces.jsonl").is_file()
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 3 * 2 * len(ARM_ORDER)
    records = [json.loads(line) for line in lines]
    assert all(record["objective_queries"] == config["budget"]
               for record in records)
    decision = json.loads((tmp_path / "frozen_decision.json").read_text())
    assert decision["baseline_arm"] in ARM_ORDER
    assert validate_portfolio_manifest(tmp_path, config)["decision_sha256"] == \
        decision["decision_sha256"]
    with pytest.raises(RuntimeError, match="audit traces already exist"):
        run_experiment(config, tmp_path, protocol)
