import json
from pathlib import Path

import pytest

from specops_gctr.risk_control import (
    binomial_lower_tail_probability,
    clopper_pearson_upper,
    evaluate_risk_gate,
    main,
    summarize_locked_audit,
    validate_prospective_design,
)


ROOT = Path(__file__).resolve().parents[1]
LOCKED_SUMMARY = (
    ROOT / "portfolio_results" / "heterogeneous_confirmatory_v1"
    / "summary.json"
)
PROSPECTIVE_DESIGN = ROOT / "configs" / \
    "prospective_risk_control_v2.design.json"


def test_clopper_pearson_reference_values():
    assert clopper_pearson_upper(19, 160) == pytest.approx(
        0.16936226467016385)
    assert clopper_pearson_upper(4, 160) == pytest.approx(
        0.05629298524006523)
    assert clopper_pearson_upper(4, 12) == pytest.approx(
        0.6091376982290366)


def test_locked_audit_distinguishes_three_risks():
    result = summarize_locked_audit(json.loads(LOCKED_SUMMARY.read_text()))
    assert result["formal_for_locked_audit"] is False
    assert result["certified_for_locked_audit"] is False
    assert result["coverage_lower_tail_probability_at_nominal"] == \
        pytest.approx(0.24863617319039014)
    assert result["coverage_failure"]["events"] == 19
    assert result["coverage_failure"][
        "passes_hypothetical_iid_gate"] is False
    assert result["joint_harm"]["events"] == 4
    assert result["joint_harm"]["passes_hypothetical_iid_gate"] is True
    assert result["conditional_harm_given_acceptance"]["total"] == 12
    assert result["conditional_harm_given_acceptance"][
        "passes_hypothetical_iid_gate"] is False
    for name in (
        "coverage_failure",
        "joint_harm",
        "conditional_harm_given_acceptance",
    ):
        assert result[name]["certified_for_locked_audit"] is False
        assert "certified" not in result[name]


def test_risk_gate_is_stronger_than_empirical_rate():
    # The point estimate is below 10%, but the high-confidence gate fails.
    result = evaluate_risk_gate(1, 20, maximum_risk=0.10)
    assert result["empirical_rate"] == pytest.approx(0.05)
    assert result["one_sided_clopper_pearson_upper"] > 0.10
    assert result["passes_hypothetical_iid_gate"] is False


def test_locked_summary_rejects_nonintegral_coverage_count():
    summary = json.loads(LOCKED_SUMMARY.read_text())
    summary["splits"]["audit"]["empirical_one_sided_coverage"] = 0.88
    with pytest.raises(
            ValueError,
            match="empirical coverage times n_graphs must be an integer"):
        summarize_locked_audit(summary)


def test_binomial_validation_and_cli(capsys):
    with pytest.raises(ValueError):
        clopper_pearson_upper(2, 1)
    with pytest.raises(ValueError):
        binomial_lower_tail_probability(
            1, 2, null_success_probability=1.1)
    assert main([str(LOCKED_SUMMARY)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["joint_harm"]["passes_hypothetical_iid_gate"] is True
    assert output["joint_harm"]["certified_for_locked_audit"] is False


def test_prospective_design_has_coherent_iid_primary_family():
    result = validate_prospective_design(
        json.loads(PROSPECTIVE_DESIGN.read_text()))
    assert result == {
        "valid": True,
        "support_points": 8,
        "arms": 17,
        "allocated_alpha": pytest.approx(0.05),
        "minimum_accepted_draws": 100,
    }


def test_prospective_design_rejects_fixed_primary_quotas():
    design = json.loads(PROSPECTIVE_DESIGN.read_text())
    design["primary_target_mixture"]["fixed_stratum_quotas"] = True
    with pytest.raises(ValueError, match="must not impose fixed stratum quotas"):
        validate_prospective_design(design)


def test_prospective_design_requires_registration_before_calibration():
    design = json.loads(PROSPECTIVE_DESIGN.read_text())
    design["registration_requirements"][
        "runner_must_require_stage_1_receipt_before_calibration"] = False
    with pytest.raises(ValueError, match="stage-1 receipt"):
        validate_prospective_design(design)
