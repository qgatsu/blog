"""99 負例サンプリング評価の乱数依存を測る。

負例セットを 1 つしか引かないと、指標のばらつきが分からず、手法間やパラメータ間の
差が「実力差」なのか「負例の引き運」なのか区別できない。ここでは同じスコア行列に
対して seed だけを変えた負例セットを複数作り、指標の分布を見る。

スコア行列の計算は seed に依存しないので 1 回だけ行い、負例の生成と採点だけを
繰り返す。

使い方:
    PYTHONPATH=src .venv/bin/python -m recommend_ml.sensitivity --min-rating 3 --n-seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataset, load_ratings
from .evaluate import evaluate_sampled, sample_negatives
from .registry import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results"

# 逆転が観測された ItemKNN の 2 設定を必ず含める
TARGETS = [
    ("ease", {"regularization": 500}),
    ("als", {"factors": 128, "regularization": 0.01, "alpha": 1.0}),
    ("item_knn", {"top_k": 10, "shrink": 0, "alpha": 0.5}),
    ("item_knn", {"top_k": 100, "shrink": 0.0, "alpha": 0.5}),
    ("user_knn", {"top_k": 50, "shrink": 100, "alpha": 0.25}),
    ("most_popular", {}),
    ("random", {"seed": 42}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="negative sampling sensitivity")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-negatives", type=int, default=99)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.data_dir)
    dataset = build_dataset(ratings, min_rating=args.min_rating)
    history = dataset.history(args.split)

    negative_sets = {
        seed: sample_negatives(dataset, args.split, args.n_negatives, seed)
        for seed in range(args.n_seeds)
    }
    print(f"負例セット {args.n_seeds} 種を生成 (n_negatives={args.n_negatives})")

    rows = []
    for name, params in TARGETS:
        model = build_model(name, **params)
        model.fit(history)
        scores = model.scores(history)
        label = f"{name} {json.dumps(params, sort_keys=True)}" if params else name

        for seed, negatives in negative_sets.items():
            metrics = evaluate_sampled(scores, dataset, args.split, negatives, ks=(10,))
            rows.append({
                "model": name,
                "label": label,
                "seed": seed,
                "hr@10": metrics["hr@10"],
                "ndcg@10": metrics["ndcg@10"],
            })
        subset = [r for r in rows if r["label"] == label]
        values = np.array([r["hr@10"] for r in subset])
        print(f"{label:52s} hr@10 mean={values.mean():.4f} sd={values.std(ddof=1):.4f} "
              f"min={values.min():.4f} max={values.max():.4f}")

    frame = pd.DataFrame(rows)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir / f"sensitivity_min{args.min_rating}.csv"
    frame.to_csv(path, index=False)

    # ItemKNN の 2 設定について、seed ごとにどちらが勝つかを数える
    knn = frame.loc[frame["model"] == "item_knn"].pivot(
        index="seed", columns="label", values="hr@10"
    )
    if knn.shape[1] == 2:
        left, right = knn.columns
        wins_left = int((knn[left] > knn[right]).sum())
        print(f"\nItemKNN の逆転チェック ({len(knn)} seed 中)")
        print(f"  {left} が上: {wins_left}")
        print(f"  {right} が上: {len(knn) - wins_left}")
        print(f"  差の平均: {(knn[left] - knn[right]).mean():+.4f}")

    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
