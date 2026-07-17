"""Frozen split specifications and graph/protocol hashing for portfolio studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import networkx as nx
import numpy as np

from .graphs import Instance, make_instance


PROTOCOL_SCHEMA_VERSION = 1
SPLIT_ORDER = ("angle_train", "development", "calibration", "audit")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def graph_edge_hash(graph: nx.Graph) -> str:
    """Hash the deterministic labelled adjacency used by the simulator."""
    nodes = sorted(int(node) for node in graph.nodes())
    edges = sorted((min(int(u), int(v)), max(int(u), int(v)))
                   for u, v in graph.edges())
    return sha256_json({"nodes": nodes, "edges": edges})


def graph_isomorphism_fingerprint(graph: nx.Graph) -> str:
    """Relabelling-invariant leakage screen, strengthened by degree metadata."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The hashes produced for graphs without.*")
        wl = nx.weisfeiler_lehman_graph_hash(graph, iterations=4)
    degrees = sorted(int(degree) for _, degree in graph.degree())
    return sha256_json({
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "degrees": degrees,
        "wl": wl,
    })


def _validate_config(config: dict) -> None:
    if config.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported portfolio config schema_version")
    families = config.get("families")
    sizes = config.get("sizes")
    counts = config.get("per_family_size")
    starts = config.get("seed_start")
    if not families or not sizes:
        raise ValueError("families and sizes must be non-empty")
    if set(counts or {}) != set(SPLIT_ORDER):
        raise ValueError(f"per_family_size must define {SPLIT_ORDER}")
    if set(starts or {}) != set(SPLIT_ORDER):
        raise ValueError(f"seed_start must define {SPLIT_ORDER}")
    if any(int(n) != n or n < 4 or n > 20 for n in sizes):
        raise ValueError("portfolio sizes must be integers in [4, 20]")
    if any(int(counts[name]) != counts[name] or counts[name] < 1
           for name in SPLIT_ORDER):
        raise ValueError("every split count must be a positive integer")
    checkpoints = config.get("checkpoints")
    if (not checkpoints or checkpoints != sorted(set(checkpoints))
            or checkpoints[0] < 1):
        raise ValueError("checkpoints must be positive, unique and increasing")
    if int(config.get("budget", -1)) != config.get("budget"):
        raise ValueError("budget must be an integer")
    if checkpoints[-1] != config["budget"]:
        raise ValueError("the final checkpoint must equal budget")
    variants = config.get("generator_parameter_ranges")
    if variants is not None and set(variants) != set(families):
        raise ValueError(
            "generator_parameter_ranges must define every configured family")


def _sample_generator_parameters(config: dict, family: str, n: int,
                                 seed: int) -> dict | None:
    ranges = config.get("generator_parameter_ranges")
    if ranges is None:
        return None
    specification = ranges[family]
    rng = np.random.default_rng(int(seed) ^ 0x5A17)
    if family == "er":
        low, high = map(float, specification["p"])
        return {"p": float(rng.uniform(low, high))}
    if family == "rr":
        valid = [int(degree) for degree in specification["degrees"]
                 if degree < n and (n * int(degree)) % 2 == 0]
        if not valid:
            raise ValueError(f"no valid regular degree for n={n}")
        return {"degree": int(rng.choice(valid))}
    if family == "ba":
        valid = [int(value) for value in specification["attachments"]
                 if 1 <= int(value) < n]
        return {"attachment": int(rng.choice(valid))}
    if family == "ws":
        valid = [int(value) for value in specification["degrees"]
                 if 2 <= int(value) < n and int(value) % 2 == 0]
        low, high = map(float, specification["rewiring"])
        return {
            "degree": int(rng.choice(valid)),
            "rewiring": float(rng.uniform(low, high)),
        }
    raise ValueError(f"continuous generator parameters unsupported for {family}")


def _make_protocol_instance(family: str, n: int, seed: int,
                            parameters: dict | None) -> Instance:
    if parameters is None:
        return make_instance(family, n, seed)
    if family == "er":
        graph = nx.gnp_random_graph(n, float(parameters["p"]), seed=seed)
    elif family == "rr":
        graph = nx.random_regular_graph(int(parameters["degree"]), n, seed=seed)
    elif family == "ba":
        graph = nx.barabasi_albert_graph(
            n, int(parameters["attachment"]), seed=seed)
    elif family == "ws":
        graph = nx.watts_strogatz_graph(
            n, int(parameters["degree"]), float(parameters["rewiring"]),
            seed=seed)
    else:
        raise ValueError(family)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    if graph.number_of_edges() == 0:
        graph.add_edge(0, 1)
    return Instance.from_networkx(graph, family=family, seed=seed)


def build_protocol(config: dict) -> dict:
    """Materialize every graph identity without running any optimizer."""
    _validate_config(config)
    split_records: dict[str, list[dict]] = {}
    seen_labelled: set[str] = set()
    # A WL collision is checked with exact isomorphism before being rejected.
    seen_invariant: dict[str, tuple[str, nx.Graph]] = {}
    for split in SPLIT_ORDER:
        records: list[dict] = []
        seed = int(config["seed_start"][split])
        count = int(config["per_family_size"][split])
        for n in config["sizes"]:
            for family in config["families"]:
                accepted = 0
                attempts = 0
                while accepted < count:
                    attempts += 1
                    if attempts > 10000:
                        raise RuntimeError(
                            f"could not obtain {count} non-isomorphic graphs "
                            f"for {split}/{family}/n={n}")
                    parameters = _sample_generator_parameters(
                        config, str(family), int(n), seed)
                    instance = _make_protocol_instance(
                        str(family), int(n), seed, parameters)
                    seed += 1
                    labelled = graph_edge_hash(instance.graph)
                    invariant = graph_isomorphism_fingerprint(instance.graph)
                    if labelled in seen_labelled:
                        continue
                    prior = seen_invariant.get(invariant)
                    if prior is not None and nx.is_isomorphic(
                            prior[1], instance.graph):
                        continue
                    seen_labelled.add(labelled)
                    seen_invariant[invariant] = (split, instance.graph.copy())
                    records.append({
                        "family": str(family),
                        "n": int(n),
                        "seed": int(instance.seed),
                        "maxcut": int(instance.maxcut),
                        "edge_hash": labelled,
                        "isomorphism_fingerprint": invariant,
                        "generator_parameters": parameters,
                    })
                    accepted += 1
        split_records[split] = records
    body = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "networkx_version": nx.__version__,
        "config_sha256": sha256_json(config),
        "splits": split_records,
    }
    body["protocol_sha256"] = sha256_json(body)
    return body


def write_frozen_protocol(config: dict, path: str | Path) -> dict:
    protocol = build_protocol(config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def load_and_verify_protocol(config: dict, path: str | Path) -> dict:
    protocol = json.loads(Path(path).read_text())
    recorded_hash = protocol.get("protocol_sha256")
    body = {key: value for key, value in protocol.items()
            if key != "protocol_sha256"}
    if recorded_hash != sha256_json(body):
        raise RuntimeError("frozen protocol hash does not match its contents")
    if protocol.get("config_sha256") != sha256_json(config):
        raise RuntimeError("portfolio config differs from the frozen protocol")
    regenerated = build_protocol(config)
    if regenerated != protocol:
        raise RuntimeError("regenerated graphs differ from the frozen protocol")
    return protocol


def instantiate_protocol_split(protocol: dict, split: str):
    if split not in SPLIT_ORDER:
        raise ValueError(f"unknown split {split!r}")
    instances = []
    for record in protocol["splits"][split]:
        instance = _make_protocol_instance(
            record["family"], record["n"], record["seed"],
            record.get("generator_parameters"))
        if graph_edge_hash(instance.graph) != record["edge_hash"]:
            raise RuntimeError("graph hash changed after protocol freeze")
        instances.append(instance)
    return instances
