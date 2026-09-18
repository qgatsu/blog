"""実験の実行入口。

使い方:
    PYTHONPATH=src .venv/bin/python -m recommend_ml.run --models random,most_popular

同じ分割・同じ評価関数を全手法に通すことが目的。手法ごとに評価コードが分岐すると
比較が壊れるため、ここを唯一の実行経路にする。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .data import build_dataset, load_ratings
from .evaluate import evaluate_full, evaluate_sampled, sample_negatives
from .registry import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens top-N ranking benchmark")
    parser.add_argument("--models", default="random,most_popular")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULT_DIR / "results.csv")
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="指定すると rating>=min を正例とする。未指定なら全評価を正例 (NCF 系の流儀)",
    )
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--ks", default="10,20,100")
    parser.add_argument("--n-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--params",
        default="{}",
        help='モデルへ渡す引数の JSON。例: \'{"top_k": 100}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = tuple(int(k) for k in args.ks.split(","))
    splits = [s for s in args.splits.split(",") if s]
    model_params = json.loads(args.params)

    ratings = load_ratings(args.data_dir)
    dataset = build_dataset(ratings, min_rating=args.min_rating)
    print(
        f"users={dataset.n_users} items={dataset.n_items} "
        f"train_interactions={dataset.train.nnz} min_rating={args.min_rating}"
    )

    negatives = sample_negatives(dataset, "test", args.n_negatives, args.seed)

    rows = []
    for name in (m for m in args.models.split(",") if m):
        for split in splits:
            history = dataset.history(split)
            model = build_model(name, **model_params)

            started = time.perf_counter()
            model.fit(history)
            fit_seconds = time.perf_counter() - started

            scores = model.scores(history)
            full = evaluate_full(scores, dataset, split, ks)
            sampled = evaluate_sampled(scores, dataset, split, negatives, ks=(10,))

            row = {
                "model": name,
                "split": split,
                "min_rating": args.min_rating,
                "fit_seconds": round(fit_seconds, 2),
                **{k: v for k, v in full.items() if k != "n_eval_users"},
                f"sampled{args.n_negatives}_hr@10": sampled["hr@10"],
                f"sampled{args.n_negatives}_ndcg@10": sampled["ndcg@10"],
                "n_eval_users": int(full["n_eval_users"]),
                "params": json.dumps(model.params, sort_keys=True),
            }
            rows.append(row)
            print(
                f"{name:14s} {split:5s} "
                + " ".join(f"{k}={row[k]:.4f}" for k in row if k.startswith(("hr@", "ndcg@", "mrr")))
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        previous = pd.read_csv(args.out)
        keys = ["model", "split", "min_rating", "params"]
        frame = (
            pd.concat([previous, frame])
            .drop_duplicates(subset=keys, keep="last")
            .reset_index(drop=True)
        )
    frame.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
