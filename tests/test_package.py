"""Unit and smoke tests for specops-gctr.

Component tests cover the mathematical contracts the manuscript relies on:
trust-region feasibility, exact query counting (seed evaluations included),
budget caps, the QAOA objective against an independent brute-force
computation, ECE sanity on synthetic residuals, monotonicity of the budget
rule, the TQA schedule, and dataset determinism. Smoke tests cover the replot
path over committed reference source data.
"""
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.stats import wilcoxon

from specops_gctr import Config
from specops_gctr.baselines import gctr_policy, heuristic_init, tqa_init
from specops_gctr.calibration import (calibration_curve,
                                      expected_calibration_error,
                                      spearman_uncertainty_error)
from specops_gctr.graphs import generate_dataset, make_instance
from specops_gctr.optimization import (QueryCounter, _project_trust_region,
                                       run_policy)
from specops_gctr.pipeline import METHOD_ORDER, _capped_score, budget_rule
from specops_gctr.plots import FIGURE_BUILDERS, build_all
from specops_gctr.qaoa import (maxcut_cost_diagonal, qaoa_expectation,
                               qaoa_expectation_sampled, qaoa_statevector)
from specops_gctr.reproduce import (
    SCHEMA_VERSION,
    _implementation_fingerprint,
    main as reproduce_main,
    run as reproduce_run,
)
from specops_gctr.tables import write_tables


def _src() -> Path:
    # committed reference source data lives at <repo>/manuscript/source_data
    # (the same files the manuscript is built from — no duplicate copy)
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "source_data",
                     parent / "manuscript" / "source_data"):
            if (cand / "meta.json").is_file():
                return cand
    raise FileNotFoundError("committed source_data not found")


def _repo() -> Path:
    return _src().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- components

def test_qaoa_expectation_matches_bruteforce():
    """<C> from the fast statevector path equals a direct dense computation."""
    inst = make_instance("er", 6, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(3):
        params = rng.uniform(0, np.pi, size=4)
        psi = qaoa_statevector(inst.C, params, inst.n)
        assert np.isclose(np.vdot(psi, psi).real, 1.0, atol=1e-10)
        direct = float(np.sum(np.abs(psi) ** 2 * inst.C))
        assert np.isclose(qaoa_expectation(inst.C, params, inst.n), direct,
                          atol=1e-10)


def test_qaoa_zero_angles_gives_mean_cut():
    """At theta=0 the state is |+>^n, so <C> = m/2 for an unweighted graph."""
    inst = make_instance("rr", 6, seed=2)
    m = inst.graph.number_of_edges()
    val = qaoa_expectation(inst.C, np.zeros(4), inst.n)
    assert np.isclose(val, m / 2.0, atol=1e-10)


def test_public_simulator_rejects_malformed_inputs():
    inst = make_instance("er", 4, seed=12)
    with pytest.raises(ValueError, match="gamma/beta pairs"):
        qaoa_statevector(inst.C, np.zeros(3), inst.n)
    with pytest.raises(ValueError, match="shots"):
        qaoa_expectation_sampled(inst.C, np.zeros(4), inst.n, shots=0)


def test_graph_utilities_accept_noninteger_node_labels():
    graph = nx.Graph()
    graph.add_edges_from([("left", "middle"), ("middle", "right")])
    diagonal = maxcut_cost_diagonal(graph)
    from specops_gctr import exact_maxcut

    assert diagonal.shape == (8,)
    assert diagonal.max() == exact_maxcut(graph) == 2
    assert make_instance("ba", 2, 0).maxcut == 1


def test_public_networkx_constructor_and_weight_boundary():
    from specops_gctr import from_networkx

    graph = nx.Graph([("left", "middle"), ("middle", "right")])
    instance = from_networkx(graph)
    assert instance.n == 3 and instance.maxcut == 2
    assert instance.C.shape == (8,)

    weighted = graph.copy()
    weighted["left"]["middle"]["weight"] = 2.0
    with pytest.raises(ValueError, match="unweighted"):
        from_networkx(weighted)
    with pytest.raises(ValueError, match="at least one edge"):
        from_networkx(nx.empty_graph(3))


def test_sampled_estimator_unbiasedish():
    inst = make_instance("ba", 6, seed=3)
    params = np.array([0.4, 0.6, 0.8, 0.3])
    exact = qaoa_expectation(inst.C, params, inst.n)
    est = qaoa_expectation_sampled(inst.C, params, inst.n, shots=20000,
                                   rng=np.random.default_rng(0))
    assert abs(est - exact) < 0.15


def test_trust_region_retraction_feasible_and_idempotent():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = 4
        center = rng.normal(size=d)
        std = rng.uniform(0.05, 2.0, size=d)
        L_inv = np.diag(1.0 / std)
        radius = rng.uniform(0.5, 3.0)
        x = center + rng.normal(size=d) * 5
        p = _project_trust_region(x, center, L_inv, radius)
        assert np.linalg.norm(L_inv @ (p - center)) <= radius + 1e-9
        inside = center + 0.1 * radius * std
        q = _project_trust_region(inside, center, L_inv, radius)
        assert np.allclose(q, inside)


def test_query_counter_counts_every_call_including_seeds():
    inst = make_instance("ws", 6, seed=4)
    counter = QueryCounter(inst.C, inst.n, inst.maxcut)
    for _ in range(7):
        counter(np.zeros(4))
    assert counter.count == 7
    # seeded policy: unreachable target forces exhaustion of the exact budget
    res = gctr_policy(inst, budget=25, mu=np.array([0.4, 0.6, 0.8, 0.3]),
                      sigma_diag=np.full(4, 0.05),
                      heuristic_angles=np.array([0.2, 0.5, 0.7, 0.2]),
                      n_gaussian_seeds=2, target=None)
    assert res["evaluations"] <= 25


def test_budget_cap_respected():
    inst = make_instance("er", 6, seed=5)
    res = gctr_policy(inst, budget=400, mu=np.array([0.4, 0.6, 0.8, 0.3]),
                      sigma_diag=np.full(4, 0.05), budget_cap=30,
                      target=inst.maxcut + 1.0)
    assert res["evaluations"] <= 30
    assert res["evaluations_used"] == res["evaluations"]
    assert res["reached_target"] is False
    assert _capped_score(res, shared_cap=400) == 400


def test_seeded_policy_never_worse_than_best_seed():
    """With a heuristic seed, the returned value is at least the seed's value."""
    inst = make_instance("er", 8, seed=6)
    heur = np.array([0.4, 0.6, 0.8, 0.3])
    heur_val = qaoa_expectation(inst.C, heur, inst.n)
    res = gctr_policy(inst, budget=50, mu=np.zeros(4),
                      sigma_diag=np.full(4, 0.05), heuristic_angles=heur,
                      target=None)
    assert res["value"] >= heur_val - 1e-9


def test_run_policy_reports_exact_value_of_returned_params():
    inst = make_instance("ba", 6, seed=7)
    res = run_policy(inst.C, inst.n, inst.maxcut, np.array([0.4, 0.6, 0.8, 0.3]),
                     budget=20)
    assert np.isclose(res["value"],
                      qaoa_expectation(inst.C, res["params"], inst.n),
                      atol=1e-10)
    assert 0.0 <= res["ratio"] <= 1.0


def test_budget_rule_monotone_and_clamped():
    cfg = Config()
    zs = np.linspace(-5, 5, 41)
    Ks, Ts = [], []
    for z in zs:
        a = budget_rule(cfg, u=z, u_med=0.0, u_iqr=1.0)
        Ks.append(a["K"]); Ts.append(a["T"])
        assert 1 <= a["K"] <= 4
        assert cfg.budget_cap_min <= a["T"] <= cfg.budget
        assert a["n_gaussian_seeds"] == max(0, a["K"] - 2)
    assert all(k2 >= k1 for k1, k2 in zip(Ks, Ks[1:]))
    assert all(t2 >= t1 for t1, t2 in zip(Ts, Ts[1:]))


def test_ece_discriminates_calibration():
    rng = np.random.default_rng(0)
    z_good = rng.standard_normal(4000)
    z_over = rng.standard_normal(4000) * 3.0   # sigma 3x too small
    z_under = rng.standard_normal(4000) / 3.0  # sigma 3x too large
    assert expected_calibration_error(z_good) < 0.03
    assert expected_calibration_error(z_over) > 0.15
    assert expected_calibration_error(z_under) > 0.15
    pred, obs, counts, N = calibration_curve(z_good)
    assert len(pred) == len(obs) == len(counts) == 10
    assert N == 4000


def test_spearman_returns_p_and_n():
    rho, p, n = spearman_uncertainty_error([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert np.isclose(rho, 1.0) and n == 5 and p < 0.05


def test_tqa_schedule_is_linear_ramp():
    """The TQA seed is the midpoint linear ramp at dt=0.75:
    [gamma_1, beta_1, gamma_2, beta_2] = [0.1875, 0.5625, 0.5625, 0.1875]."""
    inst = make_instance("er", 6, seed=8)
    res = tqa_init(inst, budget=1, target=None)
    assert res["evaluations"] == 1  # only the (counted) seed evaluation fits
    x0 = np.array([0.1875, 0.5625, 0.5625, 0.1875])
    assert np.isclose(res["value"],
                      qaoa_expectation(inst.C, x0, inst.n), atol=1e-10)


def test_heuristic_uses_supplied_angles():
    inst = make_instance("er", 6, seed=9)
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    res = heuristic_init(inst, budget=1, angles=angles, target=None)
    assert np.isclose(res["value"],
                      qaoa_expectation(inst.C, angles, inst.n), atol=1e-10)


def test_dataset_generation_deterministic():
    a = generate_dataset(8, 2, seed0=123)
    b = generate_dataset(8, 2, seed0=123)
    for x, y in zip(a, b):
        assert x.family == y.family and x.maxcut == y.maxcut
        assert sorted(x.graph.edges()) == sorted(y.graph.edges())


def test_config_defaults_match_manuscript():
    cfg = Config()
    assert cfg.p_depth == 2
    assert cfg.train_n == 14
    assert cfg.per_family_test == 12  # 48 test instances over 4 families
    assert cfg.target_frac == 0.98


# ------------------------------------------------------------------- smoke

def test_entrypoint_is_callable():
    assert callable(reproduce_main)


def test_public_api_and_single_instance_cli(capsys):
    import json
    import specops_gctr as sg
    from specops_gctr.cli import main

    instance = sg.make_instance("er", 4, 2)
    result = sg.gctr_policy(
        instance,
        budget=4,
        mu=np.array([0.4, 0.6, 0.8, 0.3]),
        sigma_diag=np.full(4, 0.05),
        target=None,
    )
    assert result["evaluations"] <= 4
    assert np.isfinite(result["value"])

    main(["--family", "er", "--n", "4", "--budget", "4", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluations"] <= 4
    assert payload["evaluations_used"] == payload["evaluations"]
    assert len(payload["parameters"]) == 4

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "specops-gctr 3.2.0" in capsys.readouterr().out


def test_trainable_policy_api_roundtrip(tmp_path):
    """The public estimator fits, predicts, searches, and reloads on CPU."""
    from specops_gctr import GCTRPolicyModel, gctr_callback_policy

    cfg = Config(
        train_n=4, p_depth=1, spectral_k=2, hidden_dim=8,
        num_layers=1, epochs=2, budget=8, budget_cap_min=4, seed=11,
    )
    train = [
        make_instance(family, 4, seed)
        for family, seed in [("er", 101), ("rr", 102),
                             ("ba", 103), ("ws", 104)]
    ]
    calibration = [make_instance("er", 5, 201),
                   make_instance("rr", 5, 202)]
    targets = np.array([
        [0.20, 0.55], [0.25, 0.50], [0.30, 0.45], [0.35, 0.40]
    ])
    fitted = GCTRPolicyModel.fit(
        train, targets, calibration, config=cfg, verbose=False)
    prediction = fitted.predict(calibration[0], budget=6)
    assert prediction.mean.shape == prediction.variance.shape == (2,)
    assert np.all(prediction.variance > 0)
    assert 1 <= prediction.allocation["K"] <= 4

    # Graph-only prediction avoids exact MaxCut and a 2**n cost diagonal. The
    # resulting policy can be sent directly to a user-supplied objective.
    graph_prediction = fitted.predict_graph(nx.cycle_graph(24), budget=6)
    assert graph_prediction.mean.shape == (2,)
    callback_result = gctr_callback_policy(
        lambda theta: -float(np.sum((theta - 0.25) ** 2)),
        budget=6,
        mu=graph_prediction.mean,
        sigma_diag=graph_prediction.variance,
        heuristic_angles=fitted.concentration,
        n_gaussian_seeds=graph_prediction.allocation["n_gaussian_seeds"],
        budget_cap=graph_prediction.allocation["T"],
    )
    assert callback_result["evaluations_used"] <= 6
    assert np.isfinite(callback_result["value"])

    with pytest.raises(ValueError, match="at least one edge"):
        fitted.predict_graph(nx.empty_graph(4))

    result = fitted.optimize(calibration[0], budget=6, seed=3)
    assert 1 <= result["evaluations"] <= 6
    assert np.isfinite(result["value"])

    checkpoint = fitted.save(tmp_path / "policy.pt")
    restored = GCTRPolicyModel.load(checkpoint)
    reloaded = restored.predict(calibration[0], budget=6)
    assert np.allclose(reloaded.mean, prediction.mean)
    assert np.allclose(reloaded.variance, prediction.variance)
    assert np.isclose(reloaded.uncertainty, prediction.uncertainty)

    with pytest.raises(ValueError, match="disjoint"):
        GCTRPolicyModel.fit(train, targets, train[:2], config=cfg)
    duplicate_with_new_metadata = type(train[0]).from_networkx(
        train[0].graph, family="renamed", seed=999)
    with pytest.raises(ValueError, match="disjoint"):
        GCTRPolicyModel.fit(
            train, targets, [duplicate_with_new_metadata, calibration[0]],
            config=cfg,
        )
    with pytest.raises(ValueError, match="at least two"):
        GCTRPolicyModel.fit(train, targets, calibration[:1], config=cfg)


def test_callback_policy_is_backend_neutral_and_budgeted():
    from specops_gctr import gctr_callback_policy

    optimum = np.array([0.2, 0.7, 0.5, 0.1])
    calls = []

    def objective(theta):
        calls.append(np.asarray(theta).copy())
        return -float(np.sum((theta - optimum) ** 2))

    result = gctr_callback_policy(
        objective,
        budget=24,
        mu=np.array([0.25, 0.65, 0.45, 0.15]),
        sigma_diag=np.full(4, 0.04),
        heuristic_angles=optimum,
        n_gaussian_seeds=2,
        target=-1e-14,
        sample_seed=5,
    )
    assert result["evaluations"] == len(calls)
    assert result["evaluations"] <= 24
    assert result["observed_value"] >= -1e-14
    assert np.allclose(result["params"], optimum)


def test_coordinate_search_deduplicates_identical_seeds():
    from specops_gctr import optimize_callback

    calls = []
    x0 = np.array([0.1, 0.2])
    result = optimize_callback(
        lambda x: calls.append(x.copy()) or -float(np.dot(x, x)),
        x0,
        budget=3,
        candidates=[x0.copy(), x0.copy()],
    )
    assert result["evaluations"] == len(calls) == 3
    assert sum(np.array_equal(call, x0) for call in calls) == 1


@pytest.mark.parametrize(
    "variance",
    [np.ones(3), np.array([1.0, 0.0, 1.0, 1.0])],
)
def test_callback_policy_rejects_invalid_gaussian_geometry(variance):
    from specops_gctr import gctr_callback_policy

    with pytest.raises(ValueError):
        gctr_callback_policy(
            lambda x: -float(np.dot(x, x)), 5, np.zeros(4), variance)


def test_replot_builds_all_figures(tmp_path):
    figures = build_all(_src(), tmp_path / "figures", formats=("png",), dpi=100)
    assert len(figures) == len(FIGURE_BUILDERS)
    for f in figures:
        assert Path(f).exists()


def test_tables_from_committed_csvs(tmp_path):
    tables = write_tables(_src(), tmp_path / "tables")
    assert len(tables) == 2
    for t in tables:
        text = Path(t).read_text()
        assert "\\toprule" in text and "\\bottomrule" in text


def test_committed_tables_match_their_source_data(tmp_path):
    """Prevent a stale hand-edited table from diverging from its CSV."""
    generated = write_tables(_src(), tmp_path / "tables")
    committed = _repo() / "manuscript" / "tables"
    for path in map(Path, generated):
        assert path.read_bytes() == (committed / path.name).read_bytes()


def test_committed_headline_statistics_are_self_consistent():
    """Recompute schema-2 capped costs, successes and paired statistics."""
    import pandas as pd

    source = _src()
    results = json.loads((source / "results.json").read_text())
    table = pd.read_csv(source / "Figure2_QueryEfficiency.csv").set_index(
        "method")
    gctr = np.asarray(results["queries"]["GCTR"]["evals"], dtype=float)

    for method in METHOD_ORDER:
        record = results["queries"][method]
        values = np.asarray(record["evals"], dtype=float)
        assert len(values) == 48
        assert record["evals_mean"] == pytest.approx(values.mean())
        assert record["evals_sd"] == pytest.approx(values.std(ddof=0))
        assert float(table.loc[method, "evaluations"]) == pytest.approx(
            values.mean(), abs=0.005)
        assert float(table.loc[method, "evaluations_sd"]) == pytest.approx(
            values.std(ddof=0), abs=0.005)
        if method != "GCTR":
            stat, pvalue = wilcoxon(gctr, values)
            stored = results["queries"]["_wilcoxon"][method]
            assert stored["w"] == pytest.approx(stat)
            assert stored["p"] == pytest.approx(pvalue)
            assert float(table.loc[method, "wilcoxon_p_vs_gctr"]) == \
                pytest.approx(pvalue, abs=5e-7)

    readouts = results["readouts"]
    assert readouts["evals::GCTR"] == pytest.approx(gctr.mean(), abs=0.05)
    assert readouts["evals::Random"] == pytest.approx(
        np.mean(results["queries"]["Random"]["evals"]), abs=0.05)
    assert readouts["reduction_vs_random"] == pytest.approx(
        1.0 - gctr.mean()
        / np.mean(results["queries"]["Random"]["evals"]), abs=5e-4)

    instance_rows = pd.read_csv(source / "QueryEfficiency_InstanceLevel.csv")
    assert len(instance_rows) == 48 * len(METHOD_ORDER)
    assert set(instance_rows["method"]) == set(METHOD_ORDER)
    for method in METHOD_ORDER:
        record = results["queries"][method]
        rows = instance_rows[instance_rows["method"] == method].sort_values(
            "instance_index")
        assert len(rows) == 48
        assert rows["capped_score"].to_numpy() == pytest.approx(record["evals"])
        assert rows["evaluations_used"].to_numpy() == pytest.approx(
            record["evaluations_used"])
        assert rows["reached_target"].astype(bool).to_list() == \
            record["reached_target"]
        expected_scores = np.where(rows["reached_target"],
                                   rows["evaluations_used"], 400)
        assert rows["capped_score"].to_numpy() == pytest.approx(expected_scores)

    lofo_rows = pd.read_csv(source / "LOFO_InstanceLevel.csv")
    assert len(lofo_rows) == 48 * 5
    assert set(lofo_rows["method"]) == {
        "GCTR", "Heuristic", "GNN point", "GCTR no trust region",
        "GCTR two-seed/full-cap",
    }
    assert "wilcoxon" in results["lofo"]
    assert results["schema_version"] == 2
    assert results["target_attainment_recorded"] is True
    assert results["score_semantics"] == \
        "capped_cost_to_shared_target"


def test_committed_provenance_matches_current_release():
    """Separate the frozen simulation generator from current display code."""
    import specops_gctr

    source = _src()
    manuscript = source.parent
    fingerprint = _implementation_fingerprint()
    results = json.loads((source / "results.json").read_text())
    meta = json.loads((source / "meta.json").read_text())
    manifest = json.loads(
        (manuscript / "figures" / "specops_gctr_manifest.json").read_text())

    for record in (results, meta):
        assert record["schema_version"] == 2
        assert record["package"] == "specops-gctr"
        assert record["package_version"] == "3.2.0"
        assert len(record["implementation_sha256"]) == 64
        assert record["implementation_sha256"] == \
            manifest["generator_implementation_sha256"]
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["source_schema_version"] == 2
    assert manifest["package"] == "specops-gctr"
    assert manifest["package_version"] == specops_gctr.__version__
    assert manifest["replot_implementation_sha256"] == fingerprint
    assert manifest["score_semantics"] == \
        "capped_cost_to_shared_target"
    assert manifest["target_attainment_recorded"] is True

    expected_sources = {
        f"source_data/{name}" for name in (
            "EDFig2_ShotNoise.csv", "EDFig3_SeedStability.csv",
            "Figure2_QueryEfficiency.csv",
            "Figure3_CalibrationAndUncertainty.csv",
            "Figure4_Generalization.csv", "Figure5_Ablation.csv",
            "Figure7_BudgetPolicy.csv", "LOFO_InstanceLevel.csv",
            "QueryEfficiency_InstanceLevel.csv", "meta.json", "results.json",
            "README.md", "generator-source-efe87496.zip",
        )
    }
    assert set(manifest["source_data_sha256"]) == expected_sources

    for relative, digest in manifest["source_data_sha256"].items():
        assert _sha256(manuscript / relative) == digest
    for relative, digest in manifest["generated_artifact_sha256"].items():
        artifact = manuscript / relative
        assert _sha256(artifact) == digest
        if artifact.suffix.lower() == ".pdf":
            payload = artifact.read_bytes()
            assert b"/CreationDate" not in payload
            assert b"/ModDate" not in payload


def test_replot_only_cli(tmp_path):
    manifest = reproduce_run([
        "--replot-only",
        "--source-data-dir", str(_src()),
        "--figures-dir", str(tmp_path / "figures"),
        "--tables-dir", str(tmp_path / "tables"),
        "--formats", "png",
        "--dpi", "100",
    ])
    assert manifest["mode"] == "replot-only"
    assert len(manifest["figures"]) == len(FIGURE_BUILDERS)
    # the console entry point returns a proper exit code
    assert reproduce_main([
        "--replot-only",
        "--source-data-dir", str(_src()),
        "--figures-dir", str(tmp_path / "figures2"),
        "--tables-dir", str(tmp_path / "tables2"),
        "--formats", "png",
        "--dpi", "100",
    ]) == 0
