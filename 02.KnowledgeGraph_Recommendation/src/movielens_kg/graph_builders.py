from __future__ import annotations

import pandas as pd


def collect_active_attribute_events(
    df: pd.DataFrame,
    event_id_column: str,
    attribute_columns: list[str],
    output_column_name: str,
) -> pd.DataFrame:
    event_frames = [
        df.loc[df[column] == 1, [event_id_column]].assign(**{output_column_name: column})
        for column in attribute_columns
    ]
    return pd.concat(event_frames, ignore_index=True)


def build_user_to_movie_edges(
    transaction_df: pd.DataFrame,
    user_attribute_columns: list[str],
    movie_attribute_columns: list[str],
    positive_interaction_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_transactions = (
        transaction_df.loc[
            transaction_df[positive_interaction_column] == 1,
            ["user_id", "movie_id", positive_interaction_column]
            + user_attribute_columns
            + movie_attribute_columns,
        ]
        .reset_index(drop=True)
        .reset_index(names="event_id")
    )

    user_events = collect_active_attribute_events(
        positive_transactions,
        event_id_column="event_id",
        attribute_columns=user_attribute_columns,
        output_column_name="source",
    )
    movie_events = collect_active_attribute_events(
        positive_transactions,
        event_id_column="event_id",
        attribute_columns=movie_attribute_columns,
        output_column_name="target",
    )

    edge_events = user_events.merge(movie_events, on="event_id", how="inner")

    edge_counts = (
        edge_events.groupby(["source", "target"], as_index=False)
        .size()
        .rename(columns={"size": "positive_event_count"})
    )
    source_totals = edge_counts.groupby("source", as_index=False)["positive_event_count"].sum()
    source_totals = source_totals.rename(
        columns={"positive_event_count": "source_total_positive_events"}
    )
    target_totals = edge_counts.groupby("target", as_index=False)["positive_event_count"].sum()
    target_totals = target_totals.rename(
        columns={"positive_event_count": "target_total_positive_events"}
    )

    global_positive_events = len(edge_events)

    edges = (
        edge_counts.merge(source_totals, on="source", how="left")
        .merge(target_totals, on="target", how="left")
        .assign(
            edge_type="user_to_movie",
            global_positive_events=global_positive_events,
        )
    )
    edges["conditional_prob"] = (
        edges["positive_event_count"] / edges["source_total_positive_events"]
    )
    edges["target_positive_rate"] = (
        edges["target_total_positive_events"] / edges["global_positive_events"]
    )
    edges["lift"] = edges["conditional_prob"] / edges["target_positive_rate"]

    return positive_transactions, edges.sort_values(
        ["positive_event_count", "lift"], ascending=[False, False]
    ).reset_index(drop=True)


def build_structural_movie_edges(
    movies_df: pd.DataFrame,
    movie_attribute_columns: list[str],
) -> pd.DataFrame:
    movie_attribute_events = collect_active_attribute_events(
        movies_df.reset_index(drop=True).reset_index(names="entity_id"),
        event_id_column="entity_id",
        attribute_columns=movie_attribute_columns,
        output_column_name="attribute",
    )
    edge_events = movie_attribute_events.merge(
        movie_attribute_events,
        on="entity_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    edge_events = edge_events.loc[edge_events["attribute_a"] < edge_events["attribute_b"]]

    edge_counts = (
        edge_events.groupby(["attribute_a", "attribute_b"], as_index=False)
        .size()
        .rename(
            columns={
                "attribute_a": "source",
                "attribute_b": "target",
                "size": "cooccurrence_count",
            }
        )
    )
    attribute_counts = movie_attribute_events.groupby("attribute", as_index=False).size()
    source_totals = attribute_counts.rename(
        columns={"attribute": "source", "size": "source_attribute_count"}
    )
    target_totals = attribute_counts.rename(
        columns={"attribute": "target", "size": "target_attribute_count"}
    )

    total_movies = len(movies_df)

    edges = (
        edge_counts.merge(source_totals, on="source", how="left")
        .merge(target_totals, on="target", how="left")
        .assign(edge_type="movie_to_movie_structural", total_entities=total_movies)
    )
    edges["source_support_rate"] = edges["cooccurrence_count"] / edges["source_attribute_count"]
    edges["target_support_rate"] = edges["cooccurrence_count"] / edges["target_attribute_count"]
    edges["jaccard"] = edges["cooccurrence_count"] / (
        edges["source_attribute_count"]
        + edges["target_attribute_count"]
        - edges["cooccurrence_count"]
    )
    edges["lift"] = (
        (edges["cooccurrence_count"] / edges["total_entities"])
        / (
            (edges["source_attribute_count"] / edges["total_entities"])
            * (edges["target_attribute_count"] / edges["total_entities"])
        )
    )
    return edges.sort_values(["cooccurrence_count", "lift"], ascending=[False, False]).reset_index(
        drop=True
    )


def build_user_preference_movie_edges(
    positive_transactions_df: pd.DataFrame,
    attribute_columns: list[str],
    edge_type: str,
    user_attribute_share_threshold: float,
) -> pd.DataFrame:
    user_positive_movie_counts = (
        positive_transactions_df.groupby("user_id", as_index=False)
        .size()
        .rename(columns={"size": "positive_movie_count"})
    )
    user_attribute_counts = (
        positive_transactions_df[["user_id"] + attribute_columns]
        .groupby("user_id", as_index=False)[attribute_columns]
        .sum()
    )
    user_attribute_matrix = user_attribute_counts.merge(
        user_positive_movie_counts,
        on="user_id",
        how="left",
    )
    for column in attribute_columns:
        user_attribute_matrix[column] = (
            user_attribute_matrix[column] / user_attribute_matrix["positive_movie_count"]
            >= user_attribute_share_threshold
        ).astype(int)

    user_attribute_matrix = user_attribute_matrix[["user_id"] + attribute_columns]
    user_attribute_events = collect_active_attribute_events(
        user_attribute_matrix,
        event_id_column="user_id",
        attribute_columns=attribute_columns,
        output_column_name="attribute",
    )
    edge_events = user_attribute_events.merge(
        user_attribute_events,
        on="user_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    edge_events = edge_events.loc[edge_events["attribute_a"] < edge_events["attribute_b"]]

    edge_counts = (
        edge_events.groupby(["attribute_a", "attribute_b"], as_index=False)
        .size()
        .rename(
            columns={
                "attribute_a": "source",
                "attribute_b": "target",
                "size": "cooccurrence_count",
            }
        )
    )
    attribute_counts = user_attribute_events.groupby("attribute", as_index=False).size()
    source_totals = attribute_counts.rename(
        columns={"attribute": "source", "size": "source_user_count"}
    )
    target_totals = attribute_counts.rename(
        columns={"attribute": "target", "size": "target_user_count"}
    )

    total_users = positive_transactions_df["user_id"].nunique()

    edges = (
        edge_counts.merge(source_totals, on="source", how="left")
        .merge(target_totals, on="target", how="left")
        .assign(edge_type=edge_type, total_entities=total_users)
    )
    edges["source_support_rate"] = edges["cooccurrence_count"] / edges["source_user_count"]
    edges["target_support_rate"] = edges["cooccurrence_count"] / edges["target_user_count"]
    edges["jaccard"] = edges["cooccurrence_count"] / (
        edges["source_user_count"] + edges["target_user_count"] - edges["cooccurrence_count"]
    )
    edges["lift"] = (
        (edges["cooccurrence_count"] / edges["total_entities"])
        / (
            (edges["source_user_count"] / edges["total_entities"])
            * (edges["target_user_count"] / edges["total_entities"])
        )
    )
    return edges.sort_values(["cooccurrence_count", "lift"], ascending=[False, False]).reset_index(
        drop=True
    )


def build_attribute_nodes(
    user_attribute_columns: list[str],
    movie_genre_columns: list[str],
    movie_decade_columns: list[str],
) -> pd.DataFrame:
    return (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "node_id": user_attribute_columns,
                        "node_type": "user_attribute",
                        "node_group": "user",
                    }
                ),
                pd.DataFrame(
                    {
                        "node_id": movie_genre_columns,
                        "node_type": "movie_genre",
                        "node_group": "movie",
                    }
                ),
                pd.DataFrame(
                    {
                        "node_id": movie_decade_columns,
                        "node_type": "movie_decade",
                        "node_group": "movie",
                    }
                ),
            ],
            ignore_index=True,
        )
        .sort_values(["node_group", "node_type", "node_id"])
        .reset_index(drop=True)
    )


def build_graph_edge_tables(
    user_to_movie_edges: pd.DataFrame,
    movie_structural_edges: pd.DataFrame,
    movie_preference_genre_edges: pd.DataFrame,
    movie_preference_decade_edges: pd.DataFrame,
    min_user_to_movie_count: int,
    min_item_to_item_count: int,
) -> dict[str, pd.DataFrame]:
    user_movie_graph_edges = (
        user_to_movie_edges.loc[
            user_to_movie_edges["positive_event_count"] >= min_user_to_movie_count,
            [
                "source",
                "target",
                "edge_type",
                "positive_event_count",
                "lift",
                "conditional_prob",
            ],
        ]
        .rename(
            columns={
                "positive_event_count": "raw_count",
                "lift": "score",
                "conditional_prob": "support_score",
            }
        )
        .assign(
            score_metric="lift",
            support_score_metric="conditional_prob",
            is_directed=True,
        )
        .reset_index(drop=True)
    )

    movie_structural_graph_edges = (
        movie_structural_edges.loc[
            movie_structural_edges["cooccurrence_count"] >= min_item_to_item_count,
            ["source", "target", "edge_type", "cooccurrence_count", "jaccard"],
        ]
        .rename(columns={"cooccurrence_count": "raw_count", "jaccard": "score"})
        .assign(
            score_metric="jaccard",
            support_score=lambda df: df["score"],
            support_score_metric="jaccard",
            is_directed=False,
        )
        .reset_index(drop=True)
    )

    movie_preference_genre_graph_edges = (
        movie_preference_genre_edges.loc[
            movie_preference_genre_edges["cooccurrence_count"] >= min_item_to_item_count,
            ["source", "target", "edge_type", "cooccurrence_count", "jaccard"],
        ]
        .rename(columns={"cooccurrence_count": "raw_count", "jaccard": "score"})
        .assign(
            score_metric="jaccard",
            support_score=lambda df: df["score"],
            support_score_metric="jaccard",
            is_directed=False,
        )
        .reset_index(drop=True)
    )

    movie_preference_decade_graph_edges = (
        movie_preference_decade_edges.loc[
            movie_preference_decade_edges["cooccurrence_count"] >= min_item_to_item_count,
            ["source", "target", "edge_type", "cooccurrence_count", "jaccard"],
        ]
        .rename(columns={"cooccurrence_count": "raw_count", "jaccard": "score"})
        .assign(
            score_metric="jaccard",
            support_score=lambda df: df["score"],
            support_score_metric="jaccard",
            is_directed=False,
        )
        .reset_index(drop=True)
    )

    movie_preference_graph_edges = pd.concat(
        [
            movie_preference_genre_graph_edges,
            movie_preference_decade_graph_edges,
        ],
        ignore_index=True,
    )

    all_graph_edges = pd.concat(
        [
            user_movie_graph_edges,
            movie_structural_graph_edges,
            movie_preference_graph_edges,
        ],
        ignore_index=True,
    )

    return {
        "user_movie_graph_edges": user_movie_graph_edges,
        "movie_structural_graph_edges": movie_structural_graph_edges,
        "movie_preference_genre_graph_edges": movie_preference_genre_graph_edges,
        "movie_preference_decade_graph_edges": movie_preference_decade_graph_edges,
        "movie_preference_graph_edges": movie_preference_graph_edges,
        "all_graph_edges": all_graph_edges,
    }
