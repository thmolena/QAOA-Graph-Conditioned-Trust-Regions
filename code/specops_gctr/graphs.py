"""MaxCut instance generation and Laplacian spectral features."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import networkx as nx

from .qaoa import maxcut_cost_diagonal

FAMILIES = ("er", "rr", "ba", "ws")


@dataclass
class Instance:
    graph: nx.Graph
    family: str
    n: int
    seed: int
    maxcut: int
    C: np.ndarray

    @classmethod
    def from_networkx(cls, graph: nx.Graph, *, family: str = "custom",
                      seed: int = 0) -> "Instance":
        """Build an unweighted MaxCut instance from a simple graph.

        Node labels are converted to contiguous integers in iteration order.
        Directed graphs, multigraphs, self-loops, and non-unit edge weights are
        rejected because the released objective and evidence are unweighted.
        """
        if (not isinstance(graph, nx.Graph) or graph.is_directed()
                or graph.is_multigraph()):
            raise TypeError("graph must be a simple undirected networkx.Graph")
        if graph.number_of_nodes() < 2:
            raise ValueError("graph must contain at least two nodes")
        if graph.number_of_edges() == 0:
            raise ValueError("graph must contain at least one edge")
        if nx.number_of_selfloops(graph):
            raise ValueError("self-loops are not supported")
        for _, _, attributes in graph.edges(data=True):
            if "weight" in attributes and float(attributes["weight"]) != 1.0:
                raise ValueError("only unweighted/unit-weight graphs are supported")
        if int(seed) != seed:
            raise ValueError("seed must be an integer")
        normalized = nx.convert_node_labels_to_integers(
            graph.copy(), ordering="default")
        n = normalized.number_of_nodes()
        return cls(
            graph=normalized,
            family=str(family),
            n=n,
            seed=int(seed),
            maxcut=exact_maxcut(normalized),
            C=maxcut_cost_diagonal(normalized),
        )


def exact_maxcut(graph: nx.Graph) -> int:
    nodes = list(graph.nodes())
    n = len(nodes)
    position = {node: i for i, node in enumerate(nodes)}
    edges = [(position[u], position[v]) for u, v in graph.edges()]
    best = 0
    for bits in product((0, 1), repeat=n):
        cut = 0
        for i, j in edges:
            cut += bits[i] != bits[j]
        if cut > best:
            best = cut
    return int(best)


def _make_graph(family: str, n: int, seed: int) -> nx.Graph:
    if family == "er":
        g = nx.gnp_random_graph(n, 0.5, seed=seed)
    elif family == "rr":
        d = 3 if n % 2 == 0 else 4
        d = min(d, n - 1)
        g = nx.random_regular_graph(d, n, seed=seed)
    elif family == "ba":
        g = nx.barabasi_albert_graph(n, min(2, n - 1), seed=seed)
    elif family == "ws":
        g = nx.watts_strogatz_graph(n, min(4, n - 1), 0.3, seed=seed)
    else:
        raise ValueError(family)
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def make_instance(family: str, n: int, seed: int) -> Instance:
    if family not in FAMILIES:
        raise ValueError(f"unknown graph family {family!r}; choose from {FAMILIES}")
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    n, seed = int(n), int(seed)
    g = _make_graph(family, n, seed)
    # guard against the (measure-zero at these sizes) edgeless draw
    if g.number_of_edges() == 0 and n >= 2:
        g.add_edge(0, 1)
    mc = exact_maxcut(g)
    C = maxcut_cost_diagonal(g)
    return Instance(graph=g, family=family, n=n, seed=seed, maxcut=mc, C=C)


def from_networkx(graph: nx.Graph, *, family: str = "custom",
                  seed: int = 0) -> Instance:
    """Public functional alias for :meth:`Instance.from_networkx`."""
    return Instance.from_networkx(graph, family=family, seed=seed)


def generate_dataset(n: int, per_family: int, seed0: int = 0):
    """Generate per_family instances for each family at size n."""
    if int(per_family) != per_family or per_family < 1:
        raise ValueError("per_family must be a positive integer")
    per_family = int(per_family)
    out = []
    s = seed0
    for fam in FAMILIES:
        for _ in range(per_family):
            out.append(make_instance(fam, n, s))
            s += 1
    return out


def laplacian_spectral_features(graph: nx.Graph, k: int = 8) -> np.ndarray:
    """Per-node features: degree + first k nontrivial Laplacian eigenvectors.

    Returns an [N, k+1] array. The sign of each eigenvector is canonicalized for
    deterministic runs when the selected eigenvalues are simple and separated.
    Repeated or nearly repeated eigenvalues retain a basis-rotation ambiguity;
    this raw-eigenvector encoding is therefore not claimed to be invariant to
    every node relabeling in degenerate eigenspaces.
    """
    if int(k) != k or k < 0:
        raise ValueError("k must be a nonnegative integer")
    k = int(k)
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("graph must contain at least one node")
    L = nx.laplacian_matrix(graph).toarray().astype(float)
    w, v = np.linalg.eigh(L)
    order = np.argsort(w)
    v = v[:, order]
    # drop the trivial constant eigenvector (index 0), take next k
    vecs = v[:, 1:1 + k]
    if vecs.shape[1] < k:
        vecs = np.pad(vecs, ((0, 0), (0, k - vecs.shape[1])))
    # sign canonicalization: make the entry of largest magnitude positive
    for c in range(vecs.shape[1]):
        col = vecs[:, c]
        j = np.argmax(np.abs(col))
        if col[j] < 0:
            vecs[:, c] = -col
    deg = np.array([d for _, d in graph.degree()], dtype=float).reshape(-1, 1)
    deg = deg / max(1.0, float(n))  # normalized degree d_v / n
    feats = np.concatenate([deg, vecs], axis=1)
    return feats.astype(np.float32)


def graph_feature_dim(k: int = 8) -> int:
    return k + 1
