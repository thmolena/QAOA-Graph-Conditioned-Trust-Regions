"""Public API for graph-conditioned trust-region QAOA optimization.

The package contains both the complete manuscript reproduction pipeline and
small composable primitives for running the search policy on a user-selected
MaxCut instance.  Importing :mod:`specops_gctr` does not start training or an
experiment.
"""

from .baselines import gctr_callback_policy, gctr_policy
from .graphs import (
    FAMILIES,
    Instance,
    exact_maxcut,
    from_networkx,
    generate_dataset,
    laplacian_spectral_features,
    make_instance,
)
from .estimator import GCTRPolicyModel, GCTRPrediction
from .optimization import (
    CallableQueryCounter,
    QueryCounter,
    coordinate_search,
    optimize_callback,
    run_policy,
)
from .pipeline import Config, budget_rule
from .qaoa import (
    best_bitstring_ratio,
    maxcut_cost_diagonal,
    qaoa_expectation,
    qaoa_expectation_sampled,
    qaoa_statevector,
)

__version__ = "3.2.0"

__all__ = [
    "FAMILIES",
    "Instance",
    "Config",
    "CallableQueryCounter",
    "GCTRPolicyModel",
    "GCTRPrediction",
    "QueryCounter",
    "best_bitstring_ratio",
    "budget_rule",
    "coordinate_search",
    "exact_maxcut",
    "from_networkx",
    "gctr_callback_policy",
    "gctr_policy",
    "generate_dataset",
    "laplacian_spectral_features",
    "make_instance",
    "maxcut_cost_diagonal",
    "qaoa_expectation",
    "qaoa_expectation_sampled",
    "qaoa_statevector",
    "optimize_callback",
    "run_policy",
    "__version__",
]
