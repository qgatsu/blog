"""MovieLens 1M の読み込みと leave-one-out 分割。

分割の方針:
  - ユーザーごとに最新の 1 件を test、その 1 つ前を valid、残りを train とする。
  - ML-1M は 77% の評価が「同一秒に入力された塊」の一部なので、timestamp だけで
    並べると同着の順序が実行環境に依存する。(timestamp, item_id) を キーに
    安定ソートすることで、分割を決定的にする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """ratings.dat をそのまま読む（前処理済み parquet ではなく生データを起点にする）。"""
    return pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=RATING_COLUMNS,
        encoding="latin-1",
    )


@dataclass
class Dataset:
    """leave-one-out 分割済みのデータ。

    train / train_valid は行=user_index, 列=item_index の 0/1 疎行列。
    valid_target / test_target は user_index -> 正解 item_index の配列。
    """

    train: sp.csr_matrix
    train_valid: sp.csr_matrix
    valid_target: np.ndarray
    test_target: np.ndarray
    user_ids: np.ndarray  # user_index -> 元の user_id
    item_ids: np.ndarray  # item_index -> 元の item_id

    @property
    def n_users(self) -> int:
        return self.train.shape[0]

    @property
    def n_items(self) -> int:
        return self.train.shape[1]

    def history(self, split: str) -> sp.csr_matrix:
        """評価時に「既に見た」として除外する履歴。"""
        return self.train if split == "valid" else self.train_valid

    def target(self, split: str) -> np.ndarray:
        return self.valid_target if split == "valid" else self.test_target


def build_dataset(
    ratings: pd.DataFrame,
    min_rating: int | None = None,
    min_interactions: int = 3,
) -> Dataset:
    """暗黙フィードバック化して leave-one-out 分割する。

    min_rating=None なら「評価した = 接触した」とみなして全件を正例にする
    (NCF 系論文の流儀)。min_rating=4 なら高評価のみを正例にする
    (Mult-VAE / EASE 系の流儀)。どちらの流儀かで文献値の比較先が変わるため、
    暗黙化の条件は必ず明示して使う。
    """
    frame = ratings
    if min_rating is not None:
        frame = frame.loc[frame["rating"] >= min_rating]

    counts = frame.groupby("user_id")["item_id"].transform("size")
    frame = frame.loc[counts >= min_interactions]

    # 同一 timestamp の塊があるため item_id を第 2 キーにして順序を決定的にする
    frame = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")

    user_ids = np.sort(frame["user_id"].unique())
    item_ids = np.sort(frame["item_id"].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

    rows = user_index.loc[frame["user_id"]].to_numpy()
    cols = item_index.loc[frame["item_id"]].to_numpy()

    # 各ユーザーのブロック内で末尾 2 件を valid / test に回す
    position_from_tail = frame.groupby("user_id").cumcount(ascending=False).to_numpy()
    is_test = position_from_tail == 0
    is_valid = position_from_tail == 1
    is_train = position_from_tail >= 2

    valid_target = np.full(len(user_ids), -1, dtype=np.int32)
    test_target = np.full(len(user_ids), -1, dtype=np.int32)
    valid_target[rows[is_valid]] = cols[is_valid]
    test_target[rows[is_test]] = cols[is_test]

    train = _to_csr(rows[is_train], cols[is_train], len(user_ids), len(item_ids))
    train_valid = _to_csr(
        rows[is_train | is_valid], cols[is_train | is_valid], len(user_ids), len(item_ids)
    )

    return Dataset(
        train=train,
        train_valid=train_valid,
        valid_target=valid_target,
        test_target=test_target,
        user_ids=user_ids,
        item_ids=item_ids,
    )


def _to_csr(rows: np.ndarray, cols: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(n_users, n_items))
    matrix.data[:] = 1.0  # 同一 user-item が重複しても 1 に潰す
    return matrix
