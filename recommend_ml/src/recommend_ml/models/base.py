"""推薦モデルの共通インターフェース。

すべてのモデルは「履歴行列を受け取って学習し、全 user × 全 item のスコア行列を
返す」形に揃える。評価側が同じ経路で順位を計算できるようにするための取り決めで、
ここを揃えないと手法間の比較に実装差が混ざる。

valid を測るときは train を、test を測るときは train+valid を履歴として渡す。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Recommender:
    name = "base"

    def fit(self, history: sp.csr_matrix) -> "Recommender":
        raise NotImplementedError

    def scores(self, history: sp.csr_matrix) -> np.ndarray:
        """(n_users, n_items) のスコア行列。履歴の除外は評価側が行う。"""
        raise NotImplementedError

    @property
    def params(self) -> dict[str, object]:
        return {}

    @property
    def model_bytes(self) -> int:
        """学習後に保持しているパラメータのバイト数。

        アイテム数に対する計算量の違い (item-item 系は O(I^2)、行列分解は
        O((U+I)d)) を実測で確かめるために、各モデルが自分のサイズを申告する。
        """
        return 0
