"""Optimizer arms, invariant selector and conformal abstention for GCTR.

This module intentionally has no oracle-target interface.  Training angle
labels are an explicitly counted offline cost; development, calibration and
audit utilities are fixed-budget anytime-regret curves.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import networkx as nx
import numpy as np
import torch

from .fixed_budget import (
    BudgetedObjective,
    FixedBudgetTrace,
    coordinate_trace,
    random_multistart_trace,
    spsa_trace,
)
from .graphs import Instance, graph_feature_dim, laplacian_spectral_features
from .model import GraphConditionedGaussian, build_dense_batch
from .optimization import QueryCounter, coordinate_search


ARM_ORDER = (
    "concentration",
    "tqa_dt_0p25",
    "tqa_dt_0p50",
    "tqa_dt_0p75",
    "tqa_dt_1p00",
    "knn",
    "random_multistart",
    "spsa",
    "legacy_gctr",
)

TQA_DT_BY_ARM = {
    "tqa_dt_0p25": 0.25,
    "tqa_dt_0p50": 0.50,
    "tqa_dt_0p75": 0.75,
    "tqa_dt_1p00": 1.00,
}

GRAPH_FEATURE_NAMES = (
    "n_scaled",
    "edges_per_vertex",
    "density",
    "degree_mean_scaled",
    "degree_sd_scaled",
    "degree_min_scaled",
    "degree_q25_scaled",
    "degree_median_scaled",
    "degree_q75_scaled",
    "degree_max_scaled",
    "mean_clustering",
    "transitivity",
    "triangles_per_vertex",
    "laplacian_lambda2_scaled",
    "laplacian_median_scaled",
    "laplacian_radius_scaled",
)


def graph_summary_features(graph: nx.Graph) -> np.ndarray:
    """Return a relabelling-invariant, fixed-length graph descriptor."""
    n = graph.number_of_nodes()
    if n < 2 or graph.number_of_edges() < 1:
        raise ValueError("graph must contain at least two nodes and one edge")
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    q25, median, q75 = np.quantile(degrees, [0.25, 0.5, 0.75])
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    triangles = sum(nx.triangles(graph).values()) / 3.0
    scale = float(n)
    values = np.asarray([
        n / 20.0,
        graph.number_of_edges() / scale,
        nx.density(graph),
        degrees.mean() / scale,
        degrees.std() / scale,
        degrees.min() / scale,
        q25 / scale,
        median / scale,
        q75 / scale,
        degrees.max() / scale,
        nx.average_clustering(graph),
        nx.transitivity(graph),
        triangles / scale,
        eigenvalues[1] / scale,
        np.median(eigenvalues) / scale,
        eigenvalues[-1] / scale,
    ], dtype=float)
    if values.shape != (len(GRAPH_FEATURE_NAMES),) or not np.all(
            np.isfinite(values)):
        raise RuntimeError("non-finite graph summary feature")
    return values


def generate_angle_label(
    instance: Instance,
    *,
    p_depth: int,
    n_starts: int,
    budget_per_start: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Generate one offline label and return its exact objective-call cost."""
    if int(p_depth) != p_depth or p_depth < 1:
        raise ValueError("p_depth must be positive")
    rng = np.random.default_rng(seed)
    best_value = -np.inf
    best_params: np.ndarray | None = None
    total_queries = 0
    for _ in range(int(n_starts)):
        counter = QueryCounter(instance.C, instance.n, instance.maxcut)
        x0 = np.empty(2 * int(p_depth), dtype=float)
        x0[0::2] = rng.uniform(0.0, np.pi, size=int(p_depth))
        x0[1::2] = rng.uniform(0.0, np.pi / 2.0, size=int(p_depth))
        params, value, _ = coordinate_search(
            counter, x0, int(budget_per_start), rng=rng)
        total_queries += counter.count
        if value > best_value:
            best_value, best_params = float(value), np.asarray(params).copy()
    assert best_params is not None
    return best_params, best_value, int(total_queries)


def _dense_inputs(instances: list[Instance], spectral_k: int):
    features = [laplacian_spectral_features(item.graph, spectral_k)
                for item in instances]
    adjacencies = [nx.to_numpy_array(item.graph, dtype=np.float32)
                   for item in instances]
    return build_dense_batch(features, adjacencies)


def fit_legacy_angle_model(
    instances: list[Instance],
    target_angles: np.ndarray,
    *,
    p_depth: int,
    spectral_k: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> GraphConditionedGaussian:
    """Fit the legacy Gaussian arm without using any evaluation split."""
    targets = np.asarray(target_angles, dtype=np.float32)
    expected = (len(instances), 2 * int(p_depth))
    if targets.shape != expected or not np.all(np.isfinite(targets)):
        raise ValueError(f"target_angles must have shape {expected}")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        X, A, M = _dense_inputs(instances, int(spectral_k))
        y = torch.tensor(targets, dtype=torch.float32)
        model = GraphConditionedGaussian(
            graph_feature_dim(int(spectral_k)),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            p_depth=int(p_depth),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            mu, logvar = model(X, A, M)
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (y - mu) ** 2 * inv_var).mean()
            # Mild variance regularization avoids a geometry dominated by the
            # clamp when the pilot has only a few dozen offline labels.
            loss = nll + 1e-3 * (logvar ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        return model
    finally:
        torch.set_num_threads(previous_threads)


def predict_legacy_gaussian(
    model: GraphConditionedGaussian,
    instances: list[Instance],
    spectral_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, A, M = _dense_inputs(instances, int(spectral_k))
    model.eval()
    with torch.no_grad():
        mean, logvar = model(X, A, M)
    return mean.numpy(), np.exp(logvar.numpy())


@dataclass
class PortfolioContext:
    p_depth: int
    concentration: np.ndarray
    training_features: np.ndarray
    training_feature_mean: np.ndarray
    training_feature_scale: np.ndarray
    training_angles: np.ndarray
    legacy_model: GraphConditionedGaussian
    spectral_k: int
    legacy_means: dict[str, np.ndarray]
    legacy_variances: dict[str, np.ndarray]

    def standardized_feature(self, instance: Instance) -> np.ndarray:
        feature = graph_summary_features(instance.graph)
        return (feature - self.training_feature_mean) / self.training_feature_scale


def instance_key(instance: Instance) -> str:
    return f"{instance.family}:n{instance.n}:s{instance.seed}"


def build_portfolio_context(
    training_instances: list[Instance],
    training_angles: np.ndarray,
    evaluation_instances: Iterable[Instance],
    *,
    p_depth: int,
    spectral_k: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> PortfolioContext:
    features = np.stack([
        graph_summary_features(item.graph) for item in training_instances])
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    model = fit_legacy_angle_model(
        training_instances,
        training_angles,
        p_depth=p_depth,
        spectral_k=spectral_k,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    evaluation = list(evaluation_instances)
    means, variances = predict_legacy_gaussian(model, evaluation, spectral_k)
    return PortfolioContext(
        p_depth=int(p_depth),
        concentration=np.median(np.asarray(training_angles, dtype=float), axis=0),
        training_features=standardized,
        training_feature_mean=feature_mean,
        training_feature_scale=feature_scale,
        training_angles=np.asarray(training_angles, dtype=float),
        legacy_model=model,
        spectral_k=int(spectral_k),
        legacy_means={instance_key(item): means[i] for i, item in enumerate(evaluation)},
        legacy_variances={instance_key(item): variances[i]
                          for i, item in enumerate(evaluation)},
    )


def _tqa_angles(p_depth: int, dt: float = 0.75) -> np.ndarray:
    grid = (np.arange(1, p_depth + 1) - 0.5) / p_depth
    angles = np.empty(2 * p_depth, dtype=float)
    angles[0::2] = grid * dt
    angles[1::2] = (1.0 - grid) * dt
    return angles


def run_portfolio_arm(
    arm: str,
    instance: Instance,
    context: PortfolioContext,
    *,
    budget: int,
    checkpoints: Iterable[int],
    seed: int,
) -> FixedBudgetTrace:
    """Run one arm under an identical target-free exact-evaluation ceiling."""
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown portfolio arm {arm!r}")
    objective = BudgetedObjective(instance.C, instance.n, int(budget))
    rng = np.random.default_rng(int(seed))
    if arm == "concentration":
        return coordinate_trace(
            objective, context.concentration, checkpoints, rng=rng)
    if arm in TQA_DT_BY_ARM:
        return coordinate_trace(
            objective,
            _tqa_angles(context.p_depth, TQA_DT_BY_ARM[arm]),
            checkpoints,
            rng=rng,
        )
    if arm == "knn":
        query = context.standardized_feature(instance)
        index = int(np.argmin(np.linalg.norm(
            context.training_features - query[None, :], axis=1)))
        return coordinate_trace(
            objective, context.training_angles[index], checkpoints, rng=rng)
    if arm == "random_multistart":
        return random_multistart_trace(
            objective, 2 * context.p_depth, checkpoints, restarts=4, rng=rng)
    if arm == "spsa":
        return spsa_trace(
            objective, context.concentration, checkpoints, rng=rng)

    key = instance_key(instance)
    mean = np.asarray(context.legacy_means[key], dtype=float)
    variance = np.maximum(
        np.asarray(context.legacy_variances[key], dtype=float), 1e-6)
    std = np.sqrt(variance)
    return coordinate_trace(
        objective,
        mean,
        checkpoints,
        candidates=[context.concentration],
        center=mean,
        L_inv=np.diag(1.0 / std),
        radius=2.0,
        step_scale=std / max(float(std.mean()), 1e-12),
        recenter_on_best=True,
        rng=rng,
    )


@dataclass
class RidgeUtilitySelector:
    """Small graph-invariant ridge model predicting per-arm AURC."""

    arms: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        utilities: np.ndarray,
        arms: Iterable[str],
        *,
        ridge: float,
    ) -> "RidgeUtilitySelector":
        X = np.asarray(features, dtype=float)
        Y = np.asarray(utilities, dtype=float)
        arm_names = tuple(arms)
        if X.ndim != 2 or Y.shape != (X.shape[0], len(arm_names)):
            raise ValueError("features/utilities have incompatible shapes")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (X - mean) / scale
        design = np.column_stack([np.ones(X.shape[0]), standardized])
        penalty = np.eye(design.shape[1]) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ Y)
        return cls(arm_names, mean, scale, coefficients, float(ridge))

    def predict(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        one = X.ndim == 1
        X = np.atleast_2d(X)
        standardized = (X - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(X.shape[0]), standardized])
        predictions = design @ self.coefficients
        return predictions[0] if one else predictions

    def choose(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.atleast_2d(self.predict(features))
        indices = np.argmin(predictions, axis=1)
        names = np.asarray([self.arms[index] for index in indices], dtype=object)
        return names, predictions


def finite_sample_upper_quantile(residuals: np.ndarray, alpha: float) -> float:
    """One-sided split-conformal quantile with finite-sample correction."""
    values = np.sort(np.asarray(residuals, dtype=float))
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("at least two finite calibration residuals are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


@dataclass(frozen=True)
class ConformalAbstainer:
    alpha: float
    margin: float
    residual_quantile: float
    baseline_arm: str | None = None

    @staticmethod
    def _baseline_names(
        count: int,
        *,
        baseline_arm: str | None,
        baseline_arms: Iterable[str] | None,
    ) -> list[str]:
        if baseline_arms is not None:
            names = list(baseline_arms)
            if baseline_arm is not None:
                raise ValueError(
                    "provide baseline_arm or baseline_arms, not both")
        elif baseline_arm is not None:
            names = [baseline_arm] * count
        else:
            raise ValueError("a baseline arm is required for every row")
        if len(names) != count:
            raise ValueError("baseline arms must match the number of rows")
        return names

    @classmethod
    def calibrate(
        cls,
        *,
        baseline_arm: str | None = None,
        baseline_arms: Iterable[str] | None = None,
        selected_arms: Iterable[str],
        predicted_utilities: np.ndarray,
        actual_utilities: np.ndarray,
        arms: Iterable[str],
        alpha: float,
        margin: float = 0.0,
    ) -> "ConformalAbstainer":
        arm_names = tuple(arms)
        arm_index = {name: index for index, name in enumerate(arm_names)}
        selected = list(selected_arms)
        predictions = np.asarray(predicted_utilities, dtype=float)
        actual = np.asarray(actual_utilities, dtype=float)
        if predictions.shape != actual.shape or predictions.shape[0] != len(selected):
            raise ValueError("calibration arrays have incompatible shapes")
        baselines = cls._baseline_names(
            len(selected),
            baseline_arm=baseline_arm,
            baseline_arms=baseline_arms,
        )
        residuals = []
        for row, (name, baseline_name) in enumerate(zip(selected, baselines)):
            selected_index = arm_index[name]
            baseline_index = arm_index[baseline_name]
            predicted_delta = (predictions[row, selected_index]
                               - predictions[row, baseline_index])
            actual_delta = (actual[row, selected_index]
                            - actual[row, baseline_index])
            residuals.append(actual_delta - predicted_delta)
        quantile = finite_sample_upper_quantile(np.asarray(residuals), alpha)
        return cls(
            alpha=float(alpha),
            margin=float(margin),
            residual_quantile=quantile,
            baseline_arm=baseline_arm,
        )

    def deploy(
        self,
        selected_arms: Iterable[str],
        predicted_utilities: np.ndarray,
        arms: Iterable[str],
        *,
        baseline_arms: Iterable[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arm_names = tuple(arms)
        arm_index = {name: index for index, name in enumerate(arm_names)}
        selected = list(selected_arms)
        predictions = np.asarray(predicted_utilities, dtype=float)
        baselines = self._baseline_names(
            len(selected),
            baseline_arm=self.baseline_arm if baseline_arms is None else None,
            baseline_arms=baseline_arms,
        )
        deployed = []
        accepted = []
        upper_bounds = []
        for row, (name, baseline_name) in enumerate(zip(selected, baselines)):
            index = arm_index[name]
            baseline_index = arm_index[baseline_name]
            predicted_delta = (predictions[row, index]
                               - predictions[row, baseline_index])
            upper = float(predicted_delta + self.residual_quantile)
            accept = name != baseline_name and upper < -self.margin
            deployed.append(name if accept else baseline_name)
            accepted.append(bool(accept))
            upper_bounds.append(upper)
        return (np.asarray(deployed, dtype=object),
                np.asarray(accepted, dtype=bool),
                np.asarray(upper_bounds, dtype=float))
