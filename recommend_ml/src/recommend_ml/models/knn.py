"""近傍法 (ItemKNN / UserKNN)。

協調フィルタリングで最も古い系統だが、再現性研究 (Dacrema et al. 2019 など) では
きちんとチューニングした近傍法が新しい手法を上回る例が繰り返し報告されている。
そのため「素朴な基準線」ではなく、真面目にハイパラを探索する対象として扱う。

実装上の要点:
  - 類似度は共起回数を正規化して作る。shrink は共起が少ないペアの類似度を
    割り引くための項で、これが無いと 1〜2 回しか共起していないアイテムが
    類似度 1.0 になり、ロングテールのノイズを拾う。
  - asymmetric cosine の alpha は人気バイアスの制御に効く。alpha=0.5 が通常の
    cosine で、小さくすると人気アイテム側の正規化が弱まる。
  - top_k で近傍を絞る。絞らないと全アイテムが薄く効いて人気順に近づく。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _similarity_from_cooccurrence(
    cooccurrence: np.ndarray,
    counts: np.ndarray,
    shrink: float,
    alpha: float,
) -> np.ndarray:
    """共起行列を asymmetric cosine で正規化する (alpha=0.5 が通常の cosine)。"""
    left = np.power(counts, alpha, dtype=np.float64)
    right = np.power(counts, 1.0 - alpha, dtype=np.float64)
    denominator = left[:, None] * right[None, :] + shrink + 1e-6
    similarity = cooccurrence / denominator
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _keep_top_k(similarity: np.ndarray, top_k: int) -> np.ndarray:
    """各行で上位 top_k 件だけ残し、他を 0 にする。"""
    if top_k >= similarity.shape[1]:
        return similarity
    threshold_index = similarity.shape[1] - top_k
    partitioned = np.argpartition(similarity, threshold_index, axis=1)
    mask = np.zeros_like(similarity, dtype=bool)
    np.put_along_axis(mask, partitioned[:, threshold_index:], True, axis=1)
    return np.where(mask, similarity, 0.0)


class ItemKNNRecommender(Recommender):
    """アイテム間類似度を使う近傍法。score = X · S。"""

    name = "item_knn"

    def __init__(self, top_k: int = 100, shrink: float = 0.0, alpha: float = 0.5) -> None:
        self.top_k = top_k
        self.shrink = shrink
        self.alpha = alpha
        self.similarity: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "ItemKNNRecommender":
        binary = history.astype(np.float32)
        cooccurrence = np.asarray((binary.T @ binary).todense(), dtype=np.float64)
        counts = np.diag(cooccurrence).copy()
        similarity = _similarity_from_cooccurrence(cooccurrence, counts, self.shrink, self.alpha)
        self.similarity = _keep_top_k(similarity, self.top_k).astype(np.float32)
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.similarity is None:
            raise RuntimeError("fit before scores")
        return np.asarray(history @ self.similarity, dtype=np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {"top_k": self.top_k, "shrink": self.shrink, "alpha": self.alpha}

    @property
    def model_bytes(self) -> int:
        return 0 if self.similarity is None else int(self.similarity.nbytes)

    @property
    def sparse_model_bytes(self) -> int:
        """top_k で 0 になった要素を捨て、疎行列で持った場合のバイト数。

        現実装は密行列のまま保持しているが、実務では top-k 近傍だけを疎行列で
        持つため、値 4 バイト + 列番号 4 バイトで見積もる。
        """
        if self.similarity is None:
            return 0
        return int(np.count_nonzero(self.similarity)) * 8



class UserKNNRecommender(Recommender):
    """ユーザー間類似度を使う近傍法。score = S · X。"""

    name = "user_knn"

    def __init__(self, top_k: int = 100, shrink: float = 0.0, alpha: float = 0.5) -> None:
        self.top_k = top_k
        self.shrink = shrink
        self.alpha = alpha
        self.similarity: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "UserKNNRecommender":
        binary = history.astype(np.float32)
        cooccurrence = np.asarray((binary @ binary.T).todense(), dtype=np.float64)
        counts = np.diag(cooccurrence).copy()
        similarity = _similarity_from_cooccurrence(cooccurrence, counts, self.shrink, self.alpha)
        self.similarity = _keep_top_k(similarity, self.top_k).astype(np.float32)
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.similarity is None:
            raise RuntimeError("fit before scores")
        return np.asarray(self.similarity @ history, dtype=np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {"top_k": self.top_k, "shrink": self.shrink, "alpha": self.alpha}

    @property
    def model_bytes(self) -> int:
        return 0 if self.similarity is None else int(self.similarity.nbytes)

    @property
    def sparse_model_bytes(self) -> int:
        """top_k で 0 になった要素を捨て、疎行列で持った場合のバイト数。

        現実装は密行列のまま保持しているが、実務では top-k 近傍だけを疎行列で
        持つため、値 4 バイト + 列番号 4 バイトで見積もる。
        """
        if self.similarity is None:
            return 0
        return int(np.count_nonzero(self.similarity)) * 8

