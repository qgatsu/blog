"""top-N ランキングの評価。

設計上の判断:
  - 全アイテムランキングを既定とする。ML-1M はアイテムが 3706 件しかなく、
    全ユーザー分のスコア行列が 100MB 未満に収まるため、近似近傍探索を挟む
    理由がない。近似を挟むとその誤差が手法間の差に混ざる。
  - 同点は平均順位で扱う。MostPopular のように未観測アイテムが同点になる
    手法があり、同点を「全て上位」扱いすると不当に有利、「全て下位」扱い
    すると不当に不利になる。
  - サンプリング評価 (99 負例) は論文値との突き合わせ用に併記する。
    サンプリングした指標は真の順位と一致しないことが知られているため、
    全アイテムランキングの代わりにはしない。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Dataset

NEG_INF = -np.inf


def rank_of_targets(
    scores: np.ndarray,
    history: sp.csr_matrix,
    target: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """正解アイテムの順位 (0 始まり、同点は平均順位) を返す。

    履歴に含まれるアイテムは候補から外す。順位は「正解より高いスコアの個数」と
    「同点の個数の半分」の和として求める。argsort より速く、同点も扱える。
    """
    ranks = np.full(len(target), np.nan, dtype=np.float64)

    for start in range(0, scores.shape[0], batch_size):
        stop = min(start + batch_size, scores.shape[0])
        block = scores[start:stop].astype(np.float64, copy=True)

        seen = history[start:stop].tocoo()
        block[seen.row, seen.col] = NEG_INF

        block_target = target[start:stop]
        valid = block_target >= 0
        if not valid.any():
            continue

        rows = np.nonzero(valid)[0]
        target_scores = block[rows, block_target[rows]][:, None]
        greater = (block[rows] > target_scores).sum(axis=1)
        ties = (block[rows] == target_scores).sum(axis=1) - 1  # 正解自身を除く
        ranks[start + rows] = greater + ties / 2.0

    return ranks


def summarize_ranks(ranks: np.ndarray, ks: tuple[int, ...] = (10, 20, 100)) -> dict[str, float]:
    """順位配列を HR@K / NDCG@K / MRR に集約する。

    正解が 1 件だけの leave-one-out なので HR@K と Recall@K は一致する。
    NDCG は IDCG=1 なので、ヒット時の 1/log2(rank+2) がそのまま値になる。
    """
    ranks = ranks[~np.isnan(ranks)]
    result: dict[str, float] = {"n_eval_users": float(len(ranks))}

    for k in ks:
        hit = ranks < k
        result[f"hr@{k}"] = float(hit.mean())
        gains = np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0)
        result[f"ndcg@{k}"] = float(gains.mean())

    result["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    return result


def evaluate_full(
    scores: np.ndarray,
    dataset: Dataset,
    split: str,
    ks: tuple[int, ...] = (10, 20, 100),
) -> dict[str, float]:
    ranks = rank_of_targets(scores, dataset.history(split), dataset.target(split))
    return summarize_ranks(ranks, ks)


def sample_negatives(
    dataset: Dataset,
    split: str,
    n_negatives: int = 99,
    seed: int = 42,
) -> np.ndarray:
    """ユーザーごとに未接触アイテムから負例を引く (user_index, n_negatives)。

    NCF 系論文と同じく「全期間を通して一度も接触していないアイテム」から引く。
    split ごとに引き直すのではなく、train/valid/test すべてを既知として扱う。
    """
    rng = np.random.default_rng(seed)
    n_items = dataset.n_items
    interacted = (dataset.train_valid.astype(bool)).tolil().rows
    negatives = np.empty((dataset.n_users, n_negatives), dtype=np.int32)

    for user in range(dataset.n_users):
        known = set(interacted[user])
        known.add(int(dataset.valid_target[user]))
        known.add(int(dataset.test_target[user]))
        drawn: set[int] = set()
        while len(drawn) < n_negatives:
            candidates = rng.integers(0, n_items, size=n_negatives)
            for candidate in candidates:
                item = int(candidate)
                if item not in known and item not in drawn:
                    drawn.add(item)
                    if len(drawn) == n_negatives:
                        break
        negatives[user] = sorted(drawn)

    return negatives


def evaluate_sampled(
    scores: np.ndarray,
    dataset: Dataset,
    split: str,
    negatives: np.ndarray,
    ks: tuple[int, ...] = (10,),
) -> dict[str, float]:
    """正解 1 件 + 負例 N 件の中だけで順位を測る (論文値との比較用)。"""
    target = dataset.target(split)
    valid = target >= 0
    rows = np.nonzero(valid)[0]

    target_scores = scores[rows, target[rows]][:, None]
    negative_scores = scores[rows[:, None], negatives[rows]]

    greater = (negative_scores > target_scores).sum(axis=1)
    ties = (negative_scores == target_scores).sum(axis=1)
    ranks = greater + ties / 2.0

    return summarize_ranks(ranks.astype(np.float64), ks)
