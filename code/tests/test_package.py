"""Unit and smoke tests for specops-gctr.

Component tests cover the mathematical contracts the manuscript relies on:
trust-region feasibility, exact query counting (seed evaluations included),
budget caps, the QAOA objective against an independent brute-force
computation, ECE sanity on synthetic residuals, monotonicity of the budget
rule, the TQA schedule, and dataset determinism. Smoke tests cover the replot
path over committed reference source data.
"""
from pathlib import Path

import numpy as np

from specops_gctr import Config
from specops_gctr.baselines import gctr_policy, heuristic_init, tqa_init
from specops_gctr.calibration import (calibration_curve,
                                      expected_calibration_error,
                                      spearman_uncertainty_error)
from specops_gctr.graphs import generate_dataset, make_instance
from specops_gctr.optimization import (QueryCounter, _project_trust_region,
                                       run_policy)
from specops_gctr.pipeline import budget_rule
from specops_gctr.plots import FIGURE_BUILDERS, build_all
from specops_gctr.qaoa import (maxcut_cost_diagonal, qaoa_expectation,
                               qaoa_expectation_sampled, qaoa_statevector)
from specops_gctr.reproduce import main as reproduce_main, run as reproduce_run
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
                      sigma_diag=np.full(4, 0.05), budget_cap=30, target=None)
    assert res["evaluations"] <= 30


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
        assert 1 <= a["K"] <= 5
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
