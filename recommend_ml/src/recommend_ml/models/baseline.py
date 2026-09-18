"""比較の基準線となる 2 つの自明な手法。

この 2 つを置かないと、ある手法の数値が良いのか悪いのか判断できない。
特に MostPopular は密なデータでは強く、多くの手法がこれを明確に上回らない。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


class RandomRecommender(Recommender):
    """一様乱数でスコアを付ける。実質的な下限。"""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, history: sp.csr_matrix) -> "RandomRecommender":
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(history.shape, dtype=np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {"seed": self.seed}


class MostPopularRecommender(Recommender):
    """学習データでの接触人数が多い順。全ユーザーに同じ順序を返す。"""

    name = "most_popular"

    def __init__(self) -> None:
        self.popularity: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "MostPopularRecommender":
        self.popularity = np.asarray(history.sum(axis=0)).ravel().astype(np.float32)
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.popularity is None:
            raise RuntimeError("fit before scores")
        return np.tile(self.popularity, (history.shape[0], 1))

    @property
    def model_bytes(self) -> int:
        return 0 if self.popularity is None else int(self.popularity.nbytes)
