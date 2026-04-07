from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import pandas as pd


def propagate_activation(
    graph: nx.DiGraph,
    seed_activation: Mapping[str, float],
    iterations: int = 2,
    reinjection_weight: float = 0.25,
    edge_probability_attr: str = "transition_prob",
    include_seed_nodes: bool = True,
    node_type_filter: str | None = None,
    normalize_output: bool = False,
) -> pd.DataFrame:
    initial_activation = {
        node: float(value)
        for node, value in seed_activation.items()
        if node in graph and float(value) > 0
    }

    if not initial_activation:
        return pd.DataFrame(
            columns=["node_id", "activation", "node_type", "node_group", "rank"]
        )

    current_activation = initial_activation.copy()

    for _ in range(iterations):
        propagated_activation = {node: 0.0 for node in graph.nodes}

        for source, source_activation in current_activation.items():
            if source_activation <= 0:
                continue

            outgoing_edges = list(graph.out_edges(source, data=True))
            if not outgoing_edges:
                propagated_activation[source] += source_activation
                continue

            distributed_mass = 0.0
            for _, target, attrs in outgoing_edges:
                edge_probability = float(attrs.get(edge_probability_attr, 0.0))
                if edge_probability <= 0:
                    continue
                mass = source_activation * edge_probability
                propagated_activation[target] += mass
                distributed_mass += mass

            residual_mass = source_activation - distributed_mass
            if residual_mass > 0:
                propagated_activation[source] += residual_mass

        next_activation = propagated_activation
        if reinjection_weight > 0:
            next_activation = {
                node: (1.0 - reinjection_weight) * propagated_activation.get(node, 0.0)
                + reinjection_weight * initial_activation.get(node, 0.0)
                for node in graph.nodes
            }

        current_activation = {
            node: value for node, value in next_activation.items() if value > 0
        }

    if normalize_output:
        total_activation = sum(current_activation.values())
        if total_activation > 0:
            current_activation = {
                node: value / total_activation for node, value in current_activation.items()
            }

    results = []
    for node, activation in current_activation.items():
        node_attrs = graph.nodes[node]
        if node_type_filter and node_attrs.get("node_type") != node_type_filter:
            continue
        if not include_seed_nodes and node in initial_activation:
            continue
        results.append(
            {
                "node_id": node,
                "activation": activation,
                "node_type": node_attrs.get("node_type"),
                "node_group": node_attrs.get("node_group"),
            }
        )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return pd.DataFrame(
            columns=["node_id", "activation", "node_type", "node_group", "rank"]
        )

    result_df = result_df.sort_values("activation", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    return result_df


def trace_activation(
    graph: nx.DiGraph,
    seed_activation: Mapping[str, float],
    iterations: int = 2,
    reinjection_weight: float = 0.25,
    edge_probability_attr: str = "transition_prob",
    include_seed_nodes: bool = True,
    node_type_filter: str | None = None,
    normalize_output: bool = False,
) -> pd.DataFrame:
    initial_activation = {
        node: float(value)
        for node, value in seed_activation.items()
        if node in graph and float(value) > 0
    }

    if not initial_activation:
        return pd.DataFrame(
            columns=["iteration", "node_id", "activation", "node_type", "node_group"]
        )

    def snapshot_df(iteration: int, activation_map: Mapping[str, float]) -> pd.DataFrame:
        rows = []
        for node, activation in activation_map.items():
            if activation <= 0:
                continue
            node_attrs = graph.nodes[node]
            if node_type_filter and node_attrs.get("node_type") != node_type_filter:
                continue
            if not include_seed_nodes and node in initial_activation:
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "node_id": node,
                    "activation": activation,
                    "node_type": node_attrs.get("node_type"),
                    "node_group": node_attrs.get("node_group"),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["iteration", "node_id", "activation", "node_type", "node_group"]
            )
        return pd.DataFrame(rows)

    history_frames = [snapshot_df(0, initial_activation)]
    current_activation = initial_activation.copy()

    for iteration in range(1, iterations + 1):
        propagated_activation = {node: 0.0 for node in graph.nodes}

        for source, source_activation in current_activation.items():
            if source_activation <= 0:
                continue

            outgoing_edges = list(graph.out_edges(source, data=True))
            if not outgoing_edges:
                propagated_activation[source] += source_activation
                continue

            distributed_mass = 0.0
            for _, target, attrs in outgoing_edges:
                edge_probability = float(attrs.get(edge_probability_attr, 0.0))
                if edge_probability <= 0:
                    continue
                mass = source_activation * edge_probability
                propagated_activation[target] += mass
                distributed_mass += mass

            residual_mass = source_activation - distributed_mass
            if residual_mass > 0:
                propagated_activation[source] += residual_mass

        next_activation = propagated_activation
        if reinjection_weight > 0:
            next_activation = {
                node: (1.0 - reinjection_weight) * propagated_activation.get(node, 0.0)
                + reinjection_weight * initial_activation.get(node, 0.0)
                for node in graph.nodes
            }

        if normalize_output:
            total_activation = sum(next_activation.values())
            if total_activation > 0:
                next_activation = {
                    node: value / total_activation for node, value in next_activation.items()
                }

        current_activation = {
            node: value for node, value in next_activation.items() if value > 0
        }
        history_frames.append(snapshot_df(iteration, current_activation))

    history_df = pd.concat(history_frames, ignore_index=True)
    if history_df.empty:
        return pd.DataFrame(
            columns=["iteration", "node_id", "activation", "node_type", "node_group"]
        )

    return history_df.sort_values(
        ["iteration", "activation", "node_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
