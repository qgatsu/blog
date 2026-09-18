"""アイテム数に対する計算コストのスケーリングを測る。

精度だけを見ると ML-1M では EASE が最良だが、EASE の学習は逆行列 1 回 (O(I^3))、
モデルは密行列 (O(I^2)) なので、アイテム数が増えると成立しなくなる。一方 ALS の
モデルは因子行列 (O((U+I)d)) でアイテム数に線形。この違いが実測の傾きとして
現れるかを確認する。

アイテムは「人気上位 N 件」で絞る。ランダムに絞るとロングテールばかりが残って
密度が変わりすぎるため、実務で候補を絞るときと同じく人気順で切る。

時間を測るので、他の重い処理と同時に実行しないこと。

使い方:
    PYTHONPATH=src .venv/bin/python -m recommend_ml.scaling
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataset, load_ratings
from .evaluate import evaluate_full
from .registry import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results"

# 傾きを取るため等比 (2 倍ずつ) に並べる
DEFAULT_SIZES = (375, 750, 1500, 3000)

# コストの比較が目的なので設定は固定する (精度の最良設定ではない)
MODELS = [
    ("most_popular", {}),
    ("item_knn", {"top_k": 100, "shrink": 0.0, "alpha": 0.5}),
    ("ease", {"regularization": 250.0}),
    ("als", {"factors": 64, "regularization": 0.1, "alpha": 1.0, "iterations": 15}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cost scaling against the number of items")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--min-rating", type=int, default=None)
    return parser.parse_args()


def subsample_by_popularity(ratings: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """接触人数の多い順に n_items 件だけ残す。"""
    popularity = ratings.groupby("item_id")["user_id"].nunique().sort_values(ascending=False)
    keep = set(popularity.head(n_items).index)
    return ratings.loc[ratings["item_id"].isin(keep)].copy()


def main() -> None:
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    ratings = load_ratings(args.data_dir)

    rows = []
    for n_items in sizes:
        subset = subsample_by_popularity(ratings, n_items)
        dataset = build_dataset(subset, min_rating=args.min_rating)
        print(f"\n=== items={dataset.n_items} users={dataset.n_users} "
              f"interactions={dataset.train_valid.nnz} ===")

        for name, params in MODELS:
            model = build_model(name, **params)
            history = dataset.train_valid

            started = time.perf_counter()
            model.fit(history)
            fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            scores = model.scores(history)
            score_seconds = time.perf_counter() - started

            metrics = evaluate_full(scores, dataset, "test", ks=(10,))
            sparse_bytes = getattr(model, "sparse_model_bytes", model.model_bytes)

            row = {
                "model": name,
                "n_items": dataset.n_items,
                "n_users": dataset.n_users,
                "n_interactions": int(dataset.train_valid.nnz),
                "fit_seconds": round(fit_seconds, 4),
                "score_seconds": round(score_seconds, 4),
                "score_us_per_user": round(score_seconds / dataset.n_users * 1e6, 1),
                "model_mb": round(model.model_bytes / 1024**2, 3),
                "sparse_model_mb": round(sparse_bytes / 1024**2, 3),
                "hr@10": round(metrics["hr@10"], 4),
                "ndcg@10": round(metrics["ndcg@10"], 4),
                "params": json.dumps(params, sort_keys=True),
            }
            rows.append(row)
            print(f"{name:14s} fit={fit_seconds:8.3f}s score={score_seconds:7.3f}s "
                  f"model={row['model_mb']:8.3f}MB hr@10={row['hr@10']:.4f}")

    frame = pd.DataFrame(rows)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / "scaling.csv"
    frame.to_csv(path, index=False)

    # log-log の傾き = べき指数。理論値 (EASE の学習 3、item-item のモデル 2、
    # 行列分解のモデル 1) と一致するかを見る。
    print("\n=== アイテム数に対するべき指数 (log-log の傾き) ===")
    print(f"{'model':14s} {'fit':>8s} {'model_size':>12s}")
    for name, _ in MODELS:
        subset = frame.loc[frame["model"] == name]
        log_items = np.log(subset["n_items"].to_numpy(dtype=float))
        exponents = {}
        for column in ("fit_seconds", "model_mb"):
            values = subset[column].to_numpy(dtype=float)
            if np.all(values > 0):
                exponents[column] = np.polyfit(log_items, np.log(values), 1)[0]
            else:
                exponents[column] = float("nan")
        print(f"{name:14s} {exponents['fit_seconds']:8.2f} {exponents['model_mb']:12.2f}")

    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
