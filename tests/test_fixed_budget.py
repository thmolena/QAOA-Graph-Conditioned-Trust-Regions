import numpy as np

from specops_gctr.fixed_budget import (
    BudgetExhausted,
    BudgetedObjective,
    coordinate_trace,
    random_multistart_trace,
    spsa_trace,
)
from specops_gctr.graphs import make_instance


def test_fixed_budget_traces_use_exact_ceiling_and_are_anytime():
    instance = make_instance("er", 6, 801)
    checkpoints = (1, 2, 4, 8)
    objective = BudgetedObjective(instance.C, instance.n, 8)
    trace = coordinate_trace(
        objective, np.array([0.4, 0.6, 0.8, 0.3]), checkpoints)
    assert trace.evaluations_used == trace.objective_queries == 8
    assert trace.shots_used == 0
    assert np.all(np.diff(trace.best_values) >= -1e-12)
    assert len(trace.best_params) == len(checkpoints)
    assert np.isfinite(trace.aurc(instance.maxcut))
    try:
        objective(np.zeros(4))
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("objective allowed a call beyond its fixed budget")


def test_spsa_and_random_multistart_consume_identical_budget():
    instance = make_instance("rr", 6, 803)
    checkpoints = (1, 4, 9)
    spsa = spsa_trace(
        BudgetedObjective(instance.C, instance.n, 9),
        np.array([0.4, 0.6, 0.8, 0.3]),
        checkpoints,
        rng=np.random.default_rng(4),
    )
    random = random_multistart_trace(
        BudgetedObjective(instance.C, instance.n, 9),
        4,
        checkpoints,
        restarts=3,
        rng=np.random.default_rng(4),
    )
    assert spsa.objective_queries == random.objective_queries == 9
    assert spsa.checkpoints == random.checkpoints == checkpoints
