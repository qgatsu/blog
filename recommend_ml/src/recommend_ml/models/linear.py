"""線形 item-item モデル (EASE)。

Steck 2019 "Embarrassingly Shallow Autoencoders"。近傍法の「類似度」を手で
決めるのではなく、自分自身を再構成する重み行列 B を閉形式で解く。
対角を 0 に固定する制約 (自分自身から自分を予測させない) がラグランジュ乗数で
解け、結果として逆行列 1 回で学習が終わる。

再現性研究で「深層手法を上回ることがある」と繰り返し挙がる手法であり、
ハイパラは正則化係数 1 つだけなので比較対象として扱いやすい。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


class EASERecommender(Recommender):
    name = "ease"

    def __init__(self, regularization: float = 250.0) -> None:
        self.regularization = regularization
        self.weights: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "EASERecommender":
        binary = history.astype(np.float32)
        gramian = np.asarray((binary.T @ binary).todense(), dtype=np.float64)
        diagonal_indices = np.diag_indices(gramian.shape[0])
        gramian[diagonal_indices] += self.regularization

        inverse = np.linalg.inv(gramian)
        weights = inverse / (-np.diag(inverse))
        weights[diagonal_indices] = 0.0
        self.weights = weights.astype(np.float32)
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit before scores")
        return np.asarray(history @ self.weights, dtype=np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {"regularization": self.regularization}

    @property
    def model_bytes(self) -> int:
        return 0 if self.weights is None else int(self.weights.nbytes)
