"""valid でのグリッドサーチ。

再現性研究が繰り返し指摘しているのは「ベースラインが弱いのは手法の差ではなく
チューニング不足」という点なので、近傍法や行列分解にも同じ探索予算を与える。
探索は valid だけで行い、選ばれた 1 設定のみを test で測る。

使い方:
    PYTHONPATH=src .venv/bin/python -m recommend_ml.tune --model item_knn \
        --grid '{"top_k": [10, 50, 100], "shrink": [0, 100], "alpha": [0.25, 0.5]}'
"""

from __future__ import annotations

import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="grid search on the validation split")
    parser.add_argument("--model", required=True)
    parser.add_argument("--grid", required=True, help="パラメータ名 -> 候補リストの JSON")
    parser.add_argument("--select-by", default="ndcg@10")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--ks", default="10,20,100")
    parser.add_argument("--n-negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def expand_grid(grid: dict[str, list]) -> list[dict]:
    names = sorted(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[n] for n in names))]


def main() -> None:
    args = parse_args()
    ks = tuple(int(k) for k in args.ks.split(","))
    combinations = expand_grid(json.loads(args.grid))

    ratings = load_ratings(args.data_dir)
    dataset = build_dataset(ratings, min_rating=args.min_rating)
    negatives = sample_negatives(dataset, "test", args.n_negatives, args.seed)
    print(f"model={args.model} combinations={len(combinations)} select_by={args.select_by}")

    rows = []
    for index, params in enumerate(combinations, start=1):
        started = time.perf_counter()
        model = build_model(args.model, **params)
        model.fit(dataset.train)
        metrics = evaluate_full(model.scores(dataset.train), dataset, "valid", ks)
        elapsed = time.perf_counter() - started
        rows.append({**params, **{k: v for k, v in metrics.items() if k != "n_eval_users"},
                     "seconds": round(elapsed, 2)})
        print(f"[{index}/{len(combinations)}] {params} -> {args.select_by}={metrics[args.select_by]:.4f}"
              f" hr@100={metrics['hr@100']:.4f} ({elapsed:.1f}s)")

    frame = pd.DataFrame(rows).sort_values(args.select_by, ascending=False).reset_index(drop=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = args.result_dir / f"tuning_{args.model}_min{args.min_rating}.csv"
    frame.to_csv(tuning_path, index=False)

    best_params = {name: frame.loc[0, name] for name in json.loads(args.grid)}
    best_params = {k: v.item() if hasattr(v, "item") else v for k, v in best_params.items()}
    print(f"\nbest on valid: {best_params} {args.select_by}={frame.loc[0, args.select_by]:.4f}")

    # 選ばれた 1 設定だけを test で測る。test は train+valid を履歴として学習し直す。
    final_rows = []
    for split in ("valid", "test"):
        history = dataset.history(split)
        model = build_model(args.model, **best_params)
        model.fit(history)
        scores = model.scores(history)
        full = evaluate_full(scores, dataset, split, ks)
        sampled = evaluate_sampled(scores, dataset, split, negatives, ks=(10,))
        final_rows.append({
            "model": args.model,
            "split": split,
            "min_rating": args.min_rating,
            "fit_seconds": None,
            **{k: v for k, v in full.items() if k != "n_eval_users"},
            f"sampled{args.n_negatives}_hr@10": sampled["hr@10"],
            f"sampled{args.n_negatives}_ndcg@10": sampled["ndcg@10"],
            "n_eval_users": int(full["n_eval_users"]),
            "params": json.dumps(best_params, sort_keys=True),
        })
        print(f"{args.model:14s} {split:5s} "
              + " ".join(f"{k}={final_rows[-1][k]:.4f}"
                         for k in final_rows[-1] if k.startswith(("hr@", "ndcg@", "mrr"))))

    results_path = args.result_dir / "results.csv"
    frame_final = pd.DataFrame(final_rows)
    if results_path.exists():
        previous = pd.read_csv(results_path)
        keys = ["model", "split", "min_rating", "params"]
        frame_final = (
            pd.concat([previous, frame_final])
            .drop_duplicates(subset=keys, keep="last")
            .reset_index(drop=True)
        )
    frame_final.to_csv(results_path, index=False)
    print(f"\nsaved -> {tuning_path}\nsaved -> {results_path}")


if __name__ == "__main__":
    main()
