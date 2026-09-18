"""行列分解 (implicit ALS / BPR-MF)。

ALS: Hu et al. 2008 の暗黙フィードバック版。全未観測を「弱い負例」として扱い、
     観測にだけ信頼度 c=1+alpha を与えて二乗誤差を最小化する。閉形式の交互解法
     なので学習率などのチューニングが要らない。

BPR: Rendle et al. 2009。二乗誤差ではなく「正例は負例より上位」という順位の
     損失を直接最適化する。top-N ランキングの評価と目的関数が揃うのが利点。

どちらも同じ「ユーザー因子 × アイテム因子」の形で、違いは損失と最適化方法だけ。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit

from .base import Recommender


class ALSRecommender(Recommender):
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "ALSRecommender":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = history.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))

        user_rows = history.tocsr()
        item_rows = history.T.tocsr()

        for _ in range(self.iterations):
            self.user_factors = _als_step(
                user_rows, self.item_factors, self.regularization, self.alpha, self.factors
            )
            self.item_factors = _als_step(
                item_rows, self.user_factors, self.regularization, self.alpha, self.factors
            )
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("fit before scores")
        return (self.user_factors @ self.item_factors.T).astype(np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
        }

    @property
    def model_bytes(self) -> int:
        if self.user_factors is None or self.item_factors is None:
            return 0
        return int(self.user_factors.nbytes + self.item_factors.nbytes)



def _als_step(
    rows: sp.csr_matrix,
    fixed_factors: np.ndarray,
    regularization: float,
    alpha: float,
    factors: int,
) -> np.ndarray:
    """片側の因子を閉形式で更新する。

    未観測を含めた全アイテムの寄与は事前計算した YtY で一括して扱い、観測が
    ある分だけ差分 (c-1) を足す。これが Hu et al. の計算量削減の要点。
    """
    gramian = fixed_factors.T @ fixed_factors
    gramian_reg = gramian + regularization * np.eye(factors)
    updated = np.zeros((rows.shape[0], factors))

    indptr, indices = rows.indptr, rows.indices
    for row in range(rows.shape[0]):
        start, stop = indptr[row], indptr[row + 1]
        if start == stop:
            continue
        observed = fixed_factors[indices[start:stop]]
        # c - 1 = alpha (二値データなので観測は全て同じ信頼度)
        left = gramian_reg + alpha * (observed.T @ observed)
        right = (1.0 + alpha) * observed.sum(axis=0)
        updated[row] = np.linalg.solve(left, right)
    return updated


@njit(cache=True)
def _bpr_epoch(
    indptr: np.ndarray,
    indices: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    sampled_users: np.ndarray,
    sampled_positions: np.ndarray,
    sampled_negatives: np.ndarray,
    learning_rate: float,
    regularization: float,
) -> None:
    """1 エポック分の SGD 更新をその場で適用する。

    (u, i, j) を引いて sigmoid(-(x_ui - x_uj)) を勾配係数に使う。負例が
    たまたま正例だった場合は引き直さず読み飛ばす (確率が低く、偏りも小さい)。
    """
    n_items = item_factors.shape[0]
    for sample in range(sampled_users.shape[0]):
        user = sampled_users[sample]
        start = indptr[user]
        stop = indptr[user + 1]
        if start == stop:
            continue
        positive = indices[start + sampled_positions[sample] % (stop - start)]
        negative = sampled_negatives[sample] % n_items

        is_positive = False
        for pointer in range(start, stop):
            if indices[pointer] == negative:
                is_positive = True
                break
        if is_positive:
            continue

        user_vector = user_factors[user]
        positive_vector = item_factors[positive]
        negative_vector = item_factors[negative]

        difference = 0.0
        for factor in range(user_vector.shape[0]):
            difference += user_vector[factor] * (positive_vector[factor] - negative_vector[factor])
        coefficient = 1.0 / (1.0 + np.exp(difference))

        for factor in range(user_vector.shape[0]):
            user_value = user_vector[factor]
            positive_value = positive_vector[factor]
            negative_value = negative_vector[factor]
            user_factors[user, factor] += learning_rate * (
                coefficient * (positive_value - negative_value) - regularization * user_value
            )
            item_factors[positive, factor] += learning_rate * (
                coefficient * user_value - regularization * positive_value
            )
            item_factors[negative, factor] += learning_rate * (
                -coefficient * user_value - regularization * negative_value
            )


class BPRRecommender(Recommender):
    name = "bpr"

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        epochs: int = 100,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, history: sp.csr_matrix) -> "BPRRecommender":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = history.shape
        self.user_factors = rng.normal(0, 0.1, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.factors))

        rows = history.tocsr()
        n_samples = rows.nnz
        for _ in range(self.epochs):
            sampled_users = rng.integers(0, n_users, n_samples).astype(np.int32)
            sampled_positions = rng.integers(0, 1 << 30, n_samples).astype(np.int64)
            sampled_negatives = rng.integers(0, n_items, n_samples).astype(np.int64)
            _bpr_epoch(
                rows.indptr.astype(np.int32),
                rows.indices.astype(np.int32),
                self.user_factors,
                self.item_factors,
                sampled_users,
                sampled_positions,
                sampled_negatives,
                self.learning_rate,
                self.regularization,
            )
        return self

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("fit before scores")
        return (self.user_factors @ self.item_factors.T).astype(np.float32)

    @property
    def params(self) -> dict[str, object]:
        return {
            "factors": self.factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "epochs": self.epochs,
        }

    @property
    def model_bytes(self) -> int:
        if self.user_factors is None or self.item_factors is None:
            return 0
        return int(self.user_factors.nbytes + self.item_factors.nbytes)

