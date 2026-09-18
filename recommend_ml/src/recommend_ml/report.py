"""results.csv を比較表にまとめる。

使い方:
    PYTHONPATH=src .venv/bin/python -m recommend_ml.report --min-rating 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "results.csv"

DISPLAY_COLUMNS = [
    "model",
    "hr@10",
    "ndcg@10",
    "hr@20",
    "hr@100",
    "mrr",
    "sampled99_hr@10",
    "sampled99_ndcg@10",
    "params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize benchmark results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--min-rating", default="all", help="3 / none / all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sort-by", default="ndcg@10")
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="既定はモデルごとに最良の 1 行だけ表示する。全設定を見たいときに指定する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.results)
    frame = frame.loc[frame["split"] == args.split]

    if args.min_rating == "none":
        frame = frame.loc[frame["min_rating"].isna()]
    elif args.min_rating != "all":
        frame = frame.loc[frame["min_rating"] == float(args.min_rating)]

    columns = [c for c in DISPLAY_COLUMNS if c in frame.columns]
    if args.min_rating == "all":
        columns = ["min_rating"] + columns

    if not args.all_params:
        # 同じモデルの複数設定が残るため、既定では最良の 1 行に絞る
        frame = (
            frame.sort_values(args.sort_by, ascending=False)
            .drop_duplicates(subset=["model", "min_rating"], keep="first")
        )

    frame = frame.sort_values(args.sort_by, ascending=False)[columns]
    formatted = frame.copy()
    for column in formatted.columns:
        if formatted[column].dtype.kind == "f":
            formatted[column] = formatted[column].map(lambda v: f"{v:.4f}")

    print(f"split={args.split} min_rating={args.min_rating} sorted_by={args.sort_by}\n")
    print(to_markdown_table(formatted))


def to_markdown_table(frame: pd.DataFrame) -> str:
    """markdown のパイプ表にする (tabulate を入れないための自前実装)。"""
    columns = [str(c) for c in frame.columns]
    rows = [[str(value) for value in row] for row in frame.astype(str).to_numpy()]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(value.ljust(w) for value, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    main()
