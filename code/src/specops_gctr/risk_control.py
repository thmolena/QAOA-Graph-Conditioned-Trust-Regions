"""Exact binomial sensitivity diagnostics for selective optimizer routing.

The portfolio-v1 experiment used fixed stratum quotas and sequential
cross-split de-duplication.  Its observations therefore do not satisfy the
independent Bernoulli model assumed by the calculations in this module.  The
functions remain useful for two narrower purposes:

* reporting an explicitly labelled iid-binomial sensitivity analysis; and
* evaluating a future protocol that independently samples its audit units.

Nothing here retroactively changes the locked portfolio-v1 decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import beta, binom


def _validate_count(events: int, total: int) -> tuple[int, int]:
    if int(events) != events or int(total) != total:
        raise ValueError("events and total must be integers")
    events = int(events)
    total = int(total)
    if total < 1 or events < 0 or events > total:
        raise ValueError("require 0 <= events <= total and total >= 1")
    return events, total


def clopper_pearson_upper(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return the exact one-sided Clopper--Pearson upper confidence bound."""
    events, total = _validate_count(events, total)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if events == total:
        return 1.0
    return float(beta.ppf(confidence, events + 1, total - events))


def binomial_lower_tail_probability(
    successes: int,
    total: int,
    *,
    null_success_probability: float,
) -> float:
    """Return ``P[X <= successes]`` for an explicitly assumed binomial model."""
    successes, total = _validate_count(successes, total)
    if not 0.0 <= null_success_probability <= 1.0:
        raise ValueError("null_success_probability must lie in [0, 1]")
    return float(binom.cdf(successes, total, null_success_probability))


def evaluate_risk_gate(
    events: int,
    total: int,
    *,
    maximum_risk: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate an event-risk gate under a hypothetical iid-binomial model.

    Passing means that the exact one-sided upper confidence bound is at most
    ``maximum_risk`` under that model.  This is stronger than comparing the
    empirical event rate with the threshold, but it is not a certification
    when the observations are not iid Bernoulli trials.
    """
    events, total = _validate_count(events, total)
    if not 0.0 < maximum_risk < 1.0:
        raise ValueError("maximum_risk must lie in (0, 1)")
    upper = clopper_pearson_upper(
        events, total, confidence=confidence)
    return {
        "events": events,
        "total": total,
        "empirical_rate": float(events / total),
        "confidence": float(confidence),
        "one_sided_clopper_pearson_upper": upper,
        "maximum_risk": float(maximum_risk),
        "passes_hypothetical_iid_gate": bool(upper <= maximum_risk),
    }


def summarize_locked_audit(
    summary: dict[str, Any],
    *,
    maximum_risk: float = 0.10,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize portfolio-v1 risks without changing its frozen decision.

    The returned values are labelled as sensitivity diagnostics because the
    locked experiment is not an iid Bernoulli sample.
    """
    audit = summary["splits"]["audit"]
    total = int(audit["n_graphs"])
    coverage = float(audit["empirical_one_sided_coverage"])
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("empirical coverage must be finite and lie in [0, 1]")
    raw_coverage_successes = coverage * total
    nearest_coverage_count = round(raw_coverage_successes)
    if not math.isclose(
            raw_coverage_successes,
            nearest_coverage_count,
            rel_tol=0.0,
            abs_tol=1e-9):
        raise ValueError(
            "empirical coverage times n_graphs must be an integer count")
    coverage_successes = int(nearest_coverage_count)
    coverage_failures = total - coverage_successes
    accepted = int(audit["accepted_count"])
    harms = int(audit["joint_harm_count"])
    if accepted < 1:
        raise ValueError("conditional risk requires at least one acceptance")
    nominal_coverage = 1.0 - float(summary["conformal"]["alpha"])
    result = {
        "diagnostic_model": "iid_binomial_sensitivity_only",
        "formal_for_locked_audit": False,
        "certified_for_locked_audit": False,
        "reason_not_formal": (
            "fixed family-by-size quotas and sequential cross-split "
            "non-isomorphism filtering do not establish iid Bernoulli trials"
        ),
        "nominal_coverage": nominal_coverage,
        "coverage_lower_tail_probability_at_nominal": (
            binomial_lower_tail_probability(
                coverage_successes,
                total,
                null_success_probability=nominal_coverage,
            )
        ),
        "coverage_failure": evaluate_risk_gate(
            coverage_failures,
            total,
            maximum_risk=maximum_risk,
            confidence=confidence,
        ),
        "joint_harm": evaluate_risk_gate(
            harms,
            total,
            maximum_risk=maximum_risk,
            confidence=confidence,
        ),
        "conditional_harm_given_acceptance": evaluate_risk_gate(
            harms,
            accepted,
            maximum_risk=maximum_risk,
            confidence=confidence,
        ),
    }
    for name in (
        "coverage_failure",
        "joint_harm",
        "conditional_harm_given_acceptance",
    ):
        result[name]["certified_for_locked_audit"] = False
    return result


def validate_prospective_design(design: dict[str, Any]) -> dict[str, Any]:
    """Validate invariants of the unregistered iid-primary design draft."""
    if design.get("status") != "design_only_not_registered_not_run":
        raise ValueError("prospective design must remain explicitly unregistered")
    registration = design["registration_requirements"]
    timeline = registration["two_stage_external_timestamp"]
    if not timeline.get("stage_1_before_primary_calibration") or \
            not timeline.get("stage_2_after_calibration_before_primary_audit"):
        raise ValueError("prospective design requires two external timestamps")
    if registration.get(
            "runner_must_require_stage_1_receipt_before_calibration") is not True:
        raise ValueError("calibration must require the stage-1 receipt")
    if registration.get(
            "runner_must_require_stage_2_receipt_before_audit") is not True:
        raise ValueError("audit must require the stage-2 receipt")

    mixture = design["primary_target_mixture"]
    if mixture.get("sampling_mode") != \
            "iid_draws_from_declared_categorical_mixture":
        raise ValueError("primary calibration and audit must use iid mixture draws")
    if mixture.get("fixed_stratum_quotas") is not False:
        raise ValueError("primary design must not impose fixed stratum quotas")
    if mixture.get("cross_split_isomorphism_rejection") is not False:
        raise ValueError("primary design must retain cross-split collisions")
    if mixture.get("duplicate_graphs") != \
            "retain_and_report_without_resampling":
        raise ValueError("primary duplicate graphs must be retained")
    support = mixture["support"]
    support_ids = [item["support_id"] for item in support]
    if len(support_ids) != 8 or len(set(support_ids)) != len(support_ids):
        raise ValueError("primary mixture must have eight unique support points")
    if not math.isclose(
            sum(float(item["mixture_weight"]) for item in support),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12):
        raise ValueError("primary mixture weights must sum to one")

    fallbacks = design["primary_policy"]["fallback_registry"]
    fallback_ids = [item["support_id"] for item in fallbacks]
    if set(fallback_ids) != set(support_ids) or \
            len(fallback_ids) != len(support_ids):
        raise ValueError("fallback registry must cover every support point once")
    for item in fallbacks:
        if item.get("registered_arm_id") is None and \
                not item.get("registration_blocker"):
            raise ValueError("unresolved fallback must block registration")

    registry = design["arm_registry"]
    arms = registry["arms"]
    arm_ids = [item["arm_id"] for item in arms]
    expected_count = int(registry["exact_arm_count"])
    expected_budget = int(registry["objective_query_ceiling_per_arm"])
    if expected_count != 17 or len(arms) != expected_count or \
            len(set(arm_ids)) != expected_count:
        raise ValueError("arm registry must enumerate exactly 17 unique arms")
    for arm in arms:
        if int(arm["objective_query_ceiling"]) != expected_budget:
            raise ValueError("every arm must use the shared query ceiling")
        if not arm.get("hyperparameters") and \
                not arm.get("registration_blocker"):
            raise ValueError(
                "every arm needs hyperparameters or a registration blocker")

    inference = design["primary_estimands_and_fwer"]
    allocations = inference["alpha_allocation"]
    endpoint_ids = [item["endpoint_id"] for item in allocations]
    if endpoint_ids != [
            "mean_paired_aurc", "joint_harm", "conditional_harm"]:
        raise ValueError("primary FWER must contain only the three named endpoints")
    allocated = sum(float(item["alpha"]) for item in allocations)
    if allocated > float(inference["fwer_budget"]) + 1e-12:
        raise ValueError("primary alpha allocation exceeds the FWER budget")
    if not math.isclose(
            allocated,
            float(inference["allocated_alpha_sum"]),
            rel_tol=0.0,
            abs_tol=1e-12):
        raise ValueError("recorded alpha sum differs from endpoint allocation")
    conditional_alpha = next(
        float(item["alpha"]) for item in allocations
        if item["endpoint_id"] == "conditional_harm")
    maximum_risk = float(inference["maximum_harm_risk"])
    if not 0.0 < maximum_risk < 1.0:
        raise ValueError("primary maximum harm risk must lie in (0, 1)")
    zero_harm_minimum = math.ceil(
        math.log(conditional_alpha) / math.log(1.0 - maximum_risk))
    if zero_harm_minimum != int(
            inference[
                "zero_harm_minimum_accepted_needed_to_make_conditional_pass_possible"
            ]):
        raise ValueError("conditional-risk feasibility count is inconsistent")
    if int(inference["minimum_accepted_draws"]) < zero_harm_minimum:
        raise ValueError("minimum acceptances cannot make the risk gate pass")

    secondary = design["secondary_descriptive_stress_policies"]
    if secondary.get("cannot_authorize_primary_claim") is not True:
        raise ValueError("secondary endpoints must not authorize the primary claim")
    execution = design["execution_status"]
    if any(bool(value) for key, value in execution.items()
           if key != "claim_authorized"):
        raise ValueError("unregistered design cannot report completed execution")
    if execution.get("claim_authorized") is not False:
        raise ValueError("unregistered design cannot authorize a claim")

    return {
        "valid": True,
        "support_points": len(support),
        "arms": len(arms),
        "allocated_alpha": allocated,
        "minimum_accepted_draws": int(inference["minimum_accepted_draws"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gctr-risk-audit",
        description=(
            "Report exact-binomial sensitivity diagnostics for a locked "
            "GCTR portfolio summary."
        ),
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--maximum-risk", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    payload = json.loads(args.summary.read_text())
    result = summarize_locked_audit(
        payload,
        maximum_risk=args.maximum_risk,
        confidence=args.confidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
