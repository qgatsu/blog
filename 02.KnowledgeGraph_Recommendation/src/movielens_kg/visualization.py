from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


DEFAULT_NODE_COLOR_MAP = {
    "user_attribute": "#2E86AB",
    "movie_genre": "#E07A5F",
    "movie_decade": "#81B29A",
}

DEFAULT_EDGE_COLOR_MAP = {
    True: "rgba(58, 99, 140, 0.60)",
    False: "rgba(120, 120, 120, 0.45)",
}


def build_networkx_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    directed: bool,
) -> nx.Graph:
    graph = nx.DiGraph() if directed else nx.Graph()

    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            row.node_id,
            node_type=row.node_type,
            node_group=row.node_group,
        )

    for row in edges_df.itertuples(index=False):
        edge_attrs = {
            "edge_type": row.edge_type,
            "raw_count": row.raw_count,
            "weight": row.score,
            "score": row.score,
            "score_metric": row.score_metric,
            "is_directed": row.is_directed,
        }
        if hasattr(row, "support_score"):
            edge_attrs["support_score"] = row.support_score
        if hasattr(row, "support_score_metric"):
            edge_attrs["support_score_metric"] = row.support_score_metric
        graph.add_edge(row.source, row.target, **edge_attrs)

    return graph


def add_minmax_display_weight(
    edges_df: pd.DataFrame,
    group_column: str = "edge_type",
    value_column: str = "score",
    output_column: str = "display_weight",
) -> pd.DataFrame:
    normalized_edges = edges_df.copy()
    normalized_edges[output_column] = normalized_edges.groupby(group_column)[value_column].transform(
        lambda s: 1.0 if s.max() == s.min() else (s - s.min()) / (s.max() - s.min())
    )
    return normalized_edges


def build_mixed_to_digraph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            row.node_id,
            node_type=row.node_type,
            node_group=row.node_group,
        )

    for row in edges_df.itertuples(index=False):
        edge_attrs = {
            "edge_type": row.edge_type,
            "raw_count": row.raw_count,
            "weight": row.score,
            "score": row.score,
            "display_weight": getattr(row, "display_weight", row.score),
            "score_metric": row.score_metric,
            "is_directed": row.is_directed,
        }
        if hasattr(row, "support_score"):
            edge_attrs["support_score"] = row.support_score
        if hasattr(row, "support_score_metric"):
            edge_attrs["support_score_metric"] = row.support_score_metric
        graph.add_edge(row.source, row.target, **edge_attrs)
        if not row.is_directed:
            graph.add_edge(row.target, row.source, **edge_attrs)

    return graph


def extract_top_weight_subgraph(
    graph: nx.Graph,
    top_n_edges: int = 30,
) -> nx.Graph:
    top_edges = sorted(
        graph.edges(data=True),
        key=lambda item: item[2]["weight"],
        reverse=True,
    )[:top_n_edges]

    subgraph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    for source, target, attrs in top_edges:
        subgraph.add_node(source, **graph.nodes[source])
        subgraph.add_node(target, **graph.nodes[target])
        subgraph.add_edge(source, target, **attrs)

    return subgraph


def extract_weight_threshold_subgraph(
    graph: nx.Graph,
    min_weight: float,
    weight_attr: str = "weight",
) -> nx.Graph:
    kept_edges = [
        (source, target, attrs)
        for source, target, attrs in graph.edges(data=True)
        if attrs[weight_attr] >= min_weight
    ]

    subgraph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    for source, target, attrs in kept_edges:
        subgraph.add_node(source, **graph.nodes[source])
        subgraph.add_node(target, **graph.nodes[target])
        subgraph.add_edge(source, target, **attrs)

    return subgraph


def add_transition_probabilities(
    graph: nx.DiGraph,
    edge_type_alpha: dict[str, float] | None = None,
    score_attr: str = "score",
    score_metric_attr: str = "score_metric",
    weight_output_attr: str = "transition_weight",
    probability_output_attr: str = "transition_prob",
) -> nx.DiGraph:
    edge_type_alpha = edge_type_alpha or {}

    for source in graph.nodes:
        outgoing_edges = list(graph.out_edges(source, data=True))
        if not outgoing_edges:
            continue

        transition_weights = []
        for _, _, attrs in outgoing_edges:
            score = attrs.get(score_attr, attrs.get("weight", 0.0))
            score_metric = attrs.get(score_metric_attr, "score")
            if score_metric == "lift":
                base_weight = max(score - 1.0, 0.0)
            else:
                base_weight = max(score, 0.0)
            alpha = edge_type_alpha.get(attrs["edge_type"], 1.0)
            transition_weight = alpha * base_weight
            attrs[weight_output_attr] = transition_weight
            transition_weights.append(transition_weight)

        total_weight = sum(transition_weights)
        if total_weight <= 0:
            uniform_probability = 1.0 / len(outgoing_edges)
            for _, _, attrs in outgoing_edges:
                attrs[probability_output_attr] = uniform_probability
            continue

        for _, _, attrs in outgoing_edges:
            attrs[probability_output_attr] = attrs[weight_output_attr] / total_weight

    return graph


def draw_graph(
    graph: nx.Graph,
    title: str,
    figsize: tuple[int, int] = (12, 8),
    edge_width_scale: float = 4.0,
    weight_attr: str = "weight",
) -> None:
    pos = nx.spring_layout(graph, seed=42, k=0.9)
    node_colors = [
        DEFAULT_NODE_COLOR_MAP.get(graph.nodes[node].get("node_type"), "#999999")
        for node in graph.nodes
    ]
    max_weight = max(
        (graph.edges[edge][weight_attr] for edge in graph.edges),
        default=1.0,
    )
    edge_widths = [
        0.6 + ((graph.edges[edge][weight_attr] / max_weight) ** 1.8) * edge_width_scale
        for edge in graph.edges
    ]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=900,
        alpha=0.9,
    )
    edge_draw_kwargs = {
        "width": edge_widths,
        "alpha": 0.45,
    }
    if graph.is_directed():
        edge_draw_kwargs["arrows"] = True
        edge_draw_kwargs["arrowsize"] = 18

    nx.draw_networkx_edges(
        graph,
        pos,
        **edge_draw_kwargs,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9)
    plt.title(title)
    plt.axis("off")
    plt.show()


def build_interactive_graph_figure(
    graph: nx.Graph,
    min_probability: float = 0.3,
):
    import plotly.graph_objects as go

    interactive_subgraph = extract_weight_threshold_subgraph(
        graph,
        min_weight=min_probability,
        weight_attr="transition_prob",
    )
    pos = nx.spring_layout(
        interactive_subgraph,
        seed=42,
        k=0.9,
        weight="transition_prob",
    )

    max_transition_prob = max(
        (attrs["transition_prob"] for _, _, attrs in interactive_subgraph.edges(data=True)),
        default=1.0,
    )

    edge_traces = []
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_text = []
    arrow_annotations = []
    seen_undirected_pairs = set()

    for source, target, attrs in interactive_subgraph.edges(data=True):
        if not attrs["is_directed"]:
            pair_key = tuple(sorted((source, target)))
            if pair_key in seen_undirected_pairs:
                continue
            seen_undirected_pairs.add(pair_key)

        x0, y0 = pos[source]
        x1, y1 = pos[target]
        width = 1.0 + 6.0 * ((attrs["transition_prob"] / max_transition_prob) ** 1.8)
        hover_text = (
            f"{source} -> {target}<br>"
            f"edge_type={attrs['edge_type']}<br>"
            f"transition_prob={attrs['transition_prob']:.3f}<br>"
            f"transition_weight={attrs.get('transition_weight', 0.0):.3f}<br>"
            f"score={attrs['score']:.3f}<br>"
            f"raw_count={attrs['raw_count']}<br>"
            f"metric={attrs['score_metric']}<br>"
            f"is_directed={attrs['is_directed']}"
        )
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={
                    "width": width,
                    "color": DEFAULT_EDGE_COLOR_MAP[attrs["is_directed"]],
                    "dash": "solid" if attrs["is_directed"] else "dot",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        edge_hover_x.append((x0 + x1) / 2)
        edge_hover_y.append((y0 + y1) / 2)
        edge_hover_text.append(hover_text)

        if attrs["is_directed"]:
            dx = x1 - x0
            dy = y1 - y0
            arrow_annotations.append(
                {
                    "x": x1,
                    "y": y1,
                    "ax": x1 - dx * 0.18,
                    "ay": y1 - dy * 0.18,
                    "xref": "x",
                    "yref": "y",
                    "axref": "x",
                    "ayref": "y",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1.2,
                    "arrowwidth": max(1.0, width * 0.45),
                    "arrowcolor": DEFAULT_EDGE_COLOR_MAP[True],
                }
            )

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        hoverinfo="text",
        hovertext=edge_hover_text,
        marker={"size": 22, "color": "rgba(0, 0, 0, 0)"},
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_text = []
    node_hover_text = []
    node_color = []

    for node, attrs in interactive_subgraph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(DEFAULT_NODE_COLOR_MAP.get(attrs.get("node_type"), "#999999"))
        node_hover_text.append(
            f"node={node}<br>node_type={attrs.get('node_type')}<br>node_group={attrs.get('node_group')}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover_text,
        marker={
            "size": 30,
            "color": node_color,
            "line": {"width": 1.5, "color": "white"},
        },
        showlegend=False,
    )

    legend_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 12, "color": color},
            name=label,
            hoverinfo="skip",
        )
        for label, color in [
            ("User Attribute", "#2E86AB"),
            ("Movie Genre", "#E07A5F"),
            ("Movie Decade", "#81B29A"),
        ]
    ] + [
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line={"width": 3, "color": DEFAULT_EDGE_COLOR_MAP[True]},
            name="Directed Edge",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line={"width": 3, "color": DEFAULT_EDGE_COLOR_MAP[False], "dash": "dot"},
            name="Undirected Edge",
            hoverinfo="skip",
        ),
    ]

    fig = go.Figure(data=edge_traces + [edge_hover_trace, node_trace] + legend_traces)
    fig.update_layout(
        title=f"Combined Attribute Graph (transition_prob >= {min_probability})",
        width=1300,
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"showgrid": False, "zeroline": False, "visible": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        annotations=arrow_annotations,
    )
    return fig
