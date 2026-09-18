"""逆切り（内側を切った後に外側を手出し）と、その手出し直後の所持牌を集計する.

定義
    外側牌 X（端距離クラス 1 = 1/9、または 2 = 2/8）を**手出し**したとき、
    それより前にそのプレイヤーが **X より内側のクラスの数牌**を切っていれば逆切りとする。
    内側牌の切りは手出し・ツモ切りを問わず、**色も問わない**。
    例: 9 の手出しなら、それ以前の自分の捨て牌に 1/9 以外の数牌があれば逆切り。
        8 の手出しなら、3〜7 のいずれかを切っていれば逆切り。
測定
    X の手出し直後の同色門前手牌に、canonical 数 y = 1..9 があるか（0/1）。
    X が高位側（8, 9）なら y は 10 - y に写して低位側と統合する。
対照
    同じクラスの外側牌の手出しのうち、内側の数牌をまだ切っていないもの。
    打牌 index 別に集計し、逆切り群の index 分布で標準化してから比べる。
内訳
    逆切りの内側牌が「同色にもある」か「他色だけ」かを分けて持つ。
    また、切られている内側牌のうち最も中心寄りのクラス別にも集計する。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

HONORS: frozenset[str] = frozenset("ESWNPFC")
SUIT_OFFSET: dict[str, int] = {"m": 0, "p": 9, "s": 18}
SUIT_INDEX: dict[str, int] = {"m": 0, "p": 1, "s": 2}

CLASSES: tuple[int, int] = (1, 2)  # 外側牌の端距離クラス（1 = 1/9, 2 = 2/8）
N_CLASS: int = len(CLASSES)
MAX_OUTER_CLASS: int = max(CLASSES)
MAX_J: int = 20  # 打牌 index のビン数
N_MASK: int = 512  # 同色 9 牌の所持を表すビットマスク

STATE_NONE: int = 0  # 逆切りなし（内側の数牌をまだ切っていない）
STATE_OTHER: int = 1  # 逆切りあり・内側牌は他色だけ
STATE_SAME: int = 2  # 逆切りあり・内側牌が同色にもある
N_STATE: int = 3
GYAKU_STATES: tuple[int, int] = (STATE_OTHER, STATE_SAME)
STATE_LABEL: tuple[str, str, str] = ("逆切りなし", "内側は他色だけ", "内側に同色あり")

MAX_INNER: int = 6  # 先行内側牌の最内クラス（0 = なし, 2..5 を使う）
N_BUCKET: int = 4  # 窓内の合計枚数を 0 / 1 / 2 / 3 以上 に丸めたバケット


def build_reverse_table() -> list[int]:
    """9bit マスクの並びを上下反転する索引表を作る（鏡像の canonical 化用）."""
    rev = [0] * N_MASK
    for m in range(N_MASK):
        r = 0
        for k in range(9):
            if m & (1 << k):
                r |= 1 << (8 - k)
        rev[m] = r
    return rev


REV_TABLE: list[int] = build_reverse_table()


def canon(tile: str) -> str:
    """赤5（末尾 r）を通常牌に正規化する."""
    return tile[:2] if (len(tile) == 3 and tile[2] == "r") else tile


def tile_index(tile: str) -> int:
    """正規化済みの牌を 0..33 の索引に写す（数牌 0-26, 字牌 27-33）."""
    if tile in HONORS:
        return 27 + "ESWNPFC".index(tile)
    return SUIT_OFFSET[tile[1]] + int(tile[0]) - 1


@dataclass
class Partial:
    """部分集計。counts は (スロット, マスク) の出現数を平坦化して持つ."""

    counts: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_J * N_STATE * N_MASK)
    )
    counts2: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_J * N_STATE * N_MASK)
    )
    inner_counts: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_INNER * MAX_J * N_MASK)
    )
    inner_counts2: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_INNER * MAX_J * N_MASK)
    )
    # 窓内（手出し牌の周辺、牌自身は除く）の合計枚数
    w1_counts: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_J * N_STATE * N_BUCKET)
    )
    w2_counts: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_J * N_STATE * N_BUCKET)
    )
    inner_w2_counts: list[int] = field(
        default_factory=lambda: [0] * (N_CLASS * 2 * MAX_INNER * MAX_J * N_BUCKET)
    )
    triggers: int = 0
    observations: int = 0
    bad_handsize: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: Partial) -> None:
        """別の部分集計を足し込む."""
        a, b = self.counts, other.counts
        for i, v in enumerate(b):
            if v:
                a[i] += v
        a2, b2 = self.counts2, other.counts2
        for i, v in enumerate(b2):
            if v:
                a2[i] += v
        ia, ib = self.inner_counts, other.inner_counts
        for i, v in enumerate(ib):
            if v:
                ia[i] += v
        ia2, ib2 = self.inner_counts2, other.inner_counts2
        for i, v in enumerate(ib2):
            if v:
                ia2[i] += v
        for mine, theirs in (
            (self.w1_counts, other.w1_counts),
            (self.w2_counts, other.w2_counts),
            (self.inner_w2_counts, other.inner_w2_counts),
        ):
            for i, v in enumerate(theirs):
                if v:
                    mine[i] += v
        self.triggers += other.triggers
        self.observations += other.observations
        self.bad_handsize += other.bad_handsize
        self.errors.extend(other.errors)


def aggregate_game(events: list[dict[str, object]], part: Partial, no_reach: bool) -> None:
    """1 半荘を頭から再生し、逆切りの成立/非成立ごとの所持を部分集計に加える."""
    hands: list[list[int]] = []
    melds: list[int] = []
    dcount: list[int] = []
    # inner_max[actor][suit] = その色で切った数牌の端距離クラスの最大（0 = まだ無い）
    inner_max: list[list[int]] = []
    reach_active = False
    counts = part.counts
    counts2 = part.counts2
    inner_counts = part.inner_counts
    inner_counts2 = part.inner_counts2
    w1_counts = part.w1_counts
    w2_counts = part.w2_counts
    inner_w2_counts = part.inner_w2_counts

    for ev in events:
        etype = cast(str, ev["type"])
        if etype == "start_kyoku":
            tehais = cast("list[list[str]]", ev["tehais"])
            hands = [[0] * 34 for _ in range(4)]
            for actor, th in enumerate(tehais):
                for t in th:
                    hands[actor][tile_index(canon(t))] += 1
            melds = [0, 0, 0, 0]
            dcount = [0, 0, 0, 0]
            inner_max = [[0, 0, 0] for _ in range(4)]
            reach_active = False
            continue
        if not hands:
            continue
        if etype == "tsumo":
            hands[cast(int, ev["actor"])][tile_index(canon(cast(str, ev["pai"])))] += 1
        elif etype == "dahai":
            actor = cast(int, ev["actor"])
            pai = canon(cast(str, ev["pai"]))
            hand = hands[actor]
            hand[tile_index(pai)] -= 1
            j = dcount[actor]
            dcount[actor] += 1
            if sum(hand) != 13 - 3 * melds[actor]:
                part.bad_handsize += 1
            if pai not in HONORS:
                val = int(pai[0])
                cls = val if val < 5 else 10 - val
                suit = pai[1]
                si = SUIT_INDEX[suit]
                if (
                    cls <= MAX_OUTER_CLASS
                    and not cast(bool, ev["tsumogiri"])
                    and not (no_reach and reach_active)
                ):
                    seen = inner_max[actor]
                    max_all = max(seen)
                    if max_all > cls:
                        state = STATE_SAME if seen[si] > cls else STATE_OTHER
                    else:
                        state = STATE_NONE
                    off = SUIT_OFFSET[suit]
                    mask = 0
                    mask2 = 0
                    for k in range(9):
                        held = hand[off + k]
                        if held:
                            mask |= 1 << k
                            if held >= 2:
                                mask2 |= 1 << k
                    if val > 5:
                        m = REV_TABLE[mask]
                        m2 = REV_TABLE[mask2]
                    else:
                        m = mask
                        m2 = mask2
                    fuuro = 1 if melds[actor] else 0
                    jbin = j if j < MAX_J else MAX_J - 1
                    ci = cls - 1
                    slot = ((ci * 2 + fuuro) * MAX_J + jbin) * N_STATE + state
                    counts[slot * N_MASK + m] += 1
                    counts2[slot * N_MASK + m2] += 1
                    # 窓内の合計枚数（手出しした牌自身は含めない）
                    w1 = 0
                    for k in range(max(1, val - 1), min(9, val + 1) + 1):
                        if k != val:
                            w1 += hand[off + k - 1]
                    w2 = 0
                    for k in range(max(1, val - 2), min(9, val + 2) + 1):
                        if k != val:
                            w2 += hand[off + k - 1]
                    w1_counts[slot * N_BUCKET + min(w1, N_BUCKET - 1)] += 1
                    w2_counts[slot * N_BUCKET + min(w2, N_BUCKET - 1)] += 1
                    part.observations += 1
                    if state != STATE_NONE:
                        part.triggers += 1
                    ibin = max_all if max_all < MAX_INNER else MAX_INNER - 1
                    islot = ((ci * 2 + fuuro) * MAX_INNER + ibin) * MAX_J + jbin
                    inner_counts[islot * N_MASK + m] += 1
                    inner_counts2[islot * N_MASK + m2] += 1
                    inner_w2_counts[islot * N_BUCKET + min(w2, N_BUCKET - 1)] += 1
                if cls > inner_max[actor][si]:
                    inner_max[actor][si] = cls
        elif etype == "reach_accepted":
            reach_active = True
        elif etype in ("pon", "chi", "daiminkan", "ankan"):
            actor = cast(int, ev["actor"])
            for t in cast("list[str]", ev["consumed"]):
                hands[actor][tile_index(canon(t))] -= 1
            melds[actor] += 1
        elif etype == "kakan":
            # consumed は既にポン済みの 3 枚。手牌から出るのは pai の 1 枚だけ。
            actor = cast(int, ev["actor"])
            hands[actor][tile_index(canon(cast(str, ev["pai"])))] -= 1
        elif etype in ("end_kyoku", "end_game"):
            hands = []


def aggregate_chunk(task: tuple[list[str], bool]) -> Partial:
    """ファイル群をまとめて集計する（並列ワーカーの入口）."""
    paths, no_reach = task
    part = Partial()
    for name in paths:
        path = Path(name)
        try:
            with path.open() as f:
                events = [json.loads(line) for line in f if line.strip()]
            aggregate_game(events, part, no_reach)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, IndexError) as err:
            part.errors.append((path.name, f"{type(err).__name__}: {err}"))
    return part


def mask_bit_table() -> npt.NDArray[np.int64]:
    """(マスク, y) -> その y を所持していれば 1 の表を作る."""
    table = np.zeros((N_MASK, 9), dtype=np.int64)
    for m in range(N_MASK):
        for k in range(9):
            if m & (1 << k):
                table[m, k] = 1
    return table


@dataclass(frozen=True)
class Summary:
    """クラスごとの集計結果."""

    n: npt.NDArray[np.int64]  # (cls, j, state)
    sums: npt.NDArray[np.int64]  # (cls, j, state, y)
    inner_n: npt.NDArray[np.int64]  # (cls, inner, j)
    inner_sum: npt.NDArray[np.int64]  # (cls, inner, j, y)


def summarize(part: Partial, menzen_only: bool, min_count: int = 1) -> Summary:
    """平坦な集計を (クラス, 打牌 index, 状態, y) の形にほどく.

    min_count が 1 なら 1 枚以上の所持、2 ならその位置を対子で持っているかを見る。
    """
    bits = mask_bit_table()
    src = part.counts if min_count == 1 else part.counts2
    isrc = part.inner_counts if min_count == 1 else part.inner_counts2
    arr = np.asarray(src, dtype=np.int64).reshape(N_CLASS, 2, MAX_J, N_STATE, N_MASK)
    inner = np.asarray(isrc, dtype=np.int64).reshape(N_CLASS, 2, MAX_INNER, MAX_J, N_MASK)
    if menzen_only:
        arr = arr[:, 0]
        inner = inner[:, 0]
    else:
        arr = arr.sum(axis=1)
        inner = inner.sum(axis=1)
    return Summary(
        n=arr.sum(axis=-1),
        sums=arr @ bits,
        inner_n=inner.sum(axis=-1),
        inner_sum=inner @ bits,
    )


def standardized_rate(
    n: npt.NDArray[np.int64],
    s: npt.NDArray[np.int64],
    weight: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """層別の所持数を、与えた層の分布で標準化した所持率に直す."""
    nf = n.astype(np.float64)
    valid = (nf > 0) & (weight > 0)
    if not valid.any():
        return np.full(9, np.nan)
    rate = np.zeros((nf.shape[0], 9), dtype=np.float64)
    rate[valid] = s.astype(np.float64)[valid] / nf[valid][:, None]
    w = weight * valid
    return (rate * w[:, None]).sum(axis=0) / w.sum()


@dataclass(frozen=True)
class ClassResult:
    """1 クラス分の所持率（巡目標準化済み）."""

    cond: npt.NDArray[np.float64]
    control: npt.NDArray[np.float64]
    other: npt.NDArray[np.float64]
    same: npt.NDArray[np.float64]
    n_cond: int
    n_ctrl: int
    n_other: int
    n_same: int


def compute_class(summary: Summary, ci: int) -> ClassResult:
    """クラス ci について、逆切り群と対照群の所持率を出す."""
    n_st = summary.n[ci]
    s_st = summary.sums[ci]
    cond_n = n_st[:, GYAKU_STATES].sum(axis=1)
    cond_s = s_st[:, GYAKU_STATES, :].sum(axis=1)
    weight = cond_n.astype(np.float64)
    return ClassResult(
        cond=cond_s.sum(axis=0) / max(cond_n.sum(), 1),
        control=standardized_rate(n_st[:, STATE_NONE], s_st[:, STATE_NONE, :], weight),
        other=standardized_rate(n_st[:, STATE_OTHER], s_st[:, STATE_OTHER, :], weight),
        same=standardized_rate(n_st[:, STATE_SAME], s_st[:, STATE_SAME, :], weight),
        n_cond=int(cond_n.sum()),
        n_ctrl=int(n_st[:, STATE_NONE].sum()),
        n_other=int(n_st[:, STATE_OTHER].sum()),
        n_same=int(n_st[:, STATE_SAME].sum()),
    )


def standardized_scalar(
    n_j: npt.NDArray[np.int64],
    hit_j: npt.NDArray[np.int64],
    weight: npt.NDArray[np.float64],
) -> float:
    """打牌 index 別の (観測数, 成立数) を、与えた巡目分布で標準化した成立率にする."""
    nf = n_j.astype(np.float64)
    valid = (nf > 0) & (weight > 0)
    if not valid.any():
        return float("nan")
    rate = np.zeros(nf.shape[0], dtype=np.float64)
    rate[valid] = hit_j[valid].astype(np.float64) / nf[valid]
    w = weight * valid
    return float((rate * w).sum() / w.sum())


def window_arrays(part: Partial, menzen_only: bool) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
]:
    """窓内枚数の集計を (クラス, ...) の形にほどく."""
    w1 = np.asarray(part.w1_counts, dtype=np.int64).reshape(
        N_CLASS, 2, MAX_J, N_STATE, N_BUCKET
    )
    w2 = np.asarray(part.w2_counts, dtype=np.int64).reshape(
        N_CLASS, 2, MAX_J, N_STATE, N_BUCKET
    )
    iw2 = np.asarray(part.inner_w2_counts, dtype=np.int64).reshape(
        N_CLASS, 2, MAX_INNER, MAX_J, N_BUCKET
    )
    if menzen_only:
        return w1[:, 0], w2[:, 0], iw2[:, 0]
    return w1.sum(axis=1), w2.sum(axis=1), iw2.sum(axis=1)


def report_window(part: Partial, menzen_only: bool) -> None:
    """窓内の合計枚数（手出しした牌自身は除く）で見た成立率を出す."""
    w1, w2, iw2 = window_arrays(part, menzen_only)
    print("\n\n# 窓内の合計枚数（手出しした牌の周辺。牌自身は含めない）")
    print("# 主指標は「窓 ±2 に 1 枚以上」")
    print("# 窓の位置数: 1/9 は ±1 で 1 箇所・±2 で 2 箇所、2/8 は ±1 で 2 箇所・±2 で 3 箇所")
    for ci, cls in enumerate(CLASSES):
        label = "1/9" if cls == 1 else "2/8"
        print(f"\n## {label} の手出し")
        print(f"{'窓':>5} {'枚数':>8} {'逆切り':>9} {'対照':>9} {'lift':>9} {'オッズ比':>9}")
        for wname, arr in (("±1", w1), ("±2", w2)):
            cond_j = arr[ci, :, GYAKU_STATES, :].sum(axis=0)  # (j, bucket)
            ctrl_j = arr[ci, :, STATE_NONE, :]  # (j, bucket)
            weight = cond_j.sum(axis=-1).astype(np.float64)
            for k in range(1, N_BUCKET):
                cond = standardized_scalar(
                    cond_j.sum(axis=-1), cond_j[:, k:].sum(axis=-1), weight
                )
                ctrl = standardized_scalar(
                    ctrl_j.sum(axis=-1), ctrl_j[:, k:].sum(axis=-1), weight
                )
                odds = (
                    (cond / (1 - cond)) / (ctrl / (1 - ctrl))
                    if 0 < cond < 1 and 0 < ctrl < 1
                    else float("nan")
                )
                name = f"{k}枚以上" if k < N_BUCKET - 1 else f"{k}枚以上"
                print(
                    f"{wname:>5} {name:>8} {cond:>9.3f} {ctrl:>9.3f} "
                    f"{cond - ctrl:>+9.3f} {odds:>9.2f}"
                )

    print("\n\n# 最内クラス別（窓 ±2、巡目標準化）")
    for ci, cls in enumerate(CLASSES):
        label = "1/9" if cls == 1 else "2/8"
        weight = iw2[ci, 2:].sum(axis=0).sum(axis=-1).astype(np.float64)
        print(f"\n## {label} の手出し")
        print(f"{'最内':>5} {'n':>12} {'1枚以上':>9} {'2枚以上':>9} {'3枚以上':>9}")
        for ibin in range(MAX_INNER):
            n_j = iw2[ci, ibin].sum(axis=-1)
            n = int(n_j.sum())
            if n < 500:
                continue
            rates = [
                standardized_scalar(n_j, iw2[ci, ibin, :, k:].sum(axis=-1), weight)
                for k in (1, 2, 3)
            ]
            name = "なし" if ibin == 0 else f"{ibin}"
            print(f"{name:>5} {n:>12,d} " + " ".join(f"{r:>9.3f}" for r in rates))

    print("\n\n# 巡目別（主指標 = 窓 ±2 に 1 枚以上）")
    print("# 巡目 = そのプレイヤーの打牌回数。n が 2,000 に満たない巡目は伏せる")
    for ci, cls in enumerate(CLASSES):
        label = "1/9" if cls == 1 else "2/8"
        cond_j = w2[ci, :, GYAKU_STATES, :].sum(axis=0)
        ctrl_j = w2[ci, :, STATE_NONE, :]
        print(f"\n## {label} の手出し")
        print(
            f"{'巡目':>4} {'逆切り':>8} {'n':>10} {'対照':>8} {'n':>10} "
            f"{'lift':>8} {'オッズ比':>8}"
        )
        for j in range(MAX_J):
            n_c = int(cond_j[j].sum())
            n_t = int(ctrl_j[j].sum())
            if n_c < 2000 or n_t < 2000:
                continue
            r_c = float(cond_j[j, 1:].sum()) / n_c
            r_t = float(ctrl_j[j, 1:].sum()) / n_t
            odds = (
                (r_c / (1 - r_c)) / (r_t / (1 - r_t))
                if 0 < r_c < 1 and 0 < r_t < 1
                else float("nan")
            )
            turn = f"{j + 1}" if j < MAX_J - 1 else f"{j + 1}+"
            print(
                f"{turn:>4} {r_c:>8.3f} {n_c:>10,d} {r_t:>8.3f} {n_t:>10,d} "
                f"{r_c - r_t:>+8.3f} {odds:>8.2f}"
            )

    print("\n\n# 巡目別 × 最内クラス（窓 ±2 に 1 枚以上）")
    for ci, cls in enumerate(CLASSES):
        label = "1/9" if cls == 1 else "2/8"
        print(f"\n## {label} の手出し")
        header = "  ".join(f"{('最内' + ('なし' if i == 0 else str(i))):>8}" for i in range(MAX_INNER))
        print(f"{'巡目':>4}  {header}")
        for j in range(MAX_J):
            cells: list[str] = []
            shown = False
            for ibin in range(MAX_INNER):
                n = int(iw2[ci, ibin, j].sum())
                if n < 2000:
                    cells.append(f"{'-':>8}")
                    continue
                shown = True
                cells.append(f"{float(iw2[ci, ibin, j, 1:].sum()) / n:>8.3f}")
            if shown:
                turn = f"{j + 1}" if j < MAX_J - 1 else f"{j + 1}+"
                print(f"{turn:>4}  " + "  ".join(cells))


def report(part: Partial, menzen_only: bool, n_files: int, out_csv: Path | None) -> None:
    """集計結果を表として出力する（1 枚以上と対子の両方）."""
    s1 = summarize(part, menzen_only, 1)
    s2 = summarize(part, menzen_only, 2)
    scope = "門前のみ" if menzen_only else "副露込み"
    print(f"# 逆切り  files={n_files}  scope={scope}")
    print(
        f"# 観測 {part.observations:,}  逆切り {part.triggers:,}  "
        f"handsize不整合 {part.bad_handsize:,}  errors {len(part.errors)}"
    )
    rows: list[list[float]] = []
    for ci, cls in enumerate(CLASSES):
        r1 = compute_class(s1, ci)
        r2 = compute_class(s2, ci)
        label = "1/9" if cls == 1 else "2/8"
        print(
            f"\n## {label} の手出し   逆切り n={r1.n_cond:,}  対照 n={r1.n_ctrl:,}"
            f"（内訳 他色だけ {r1.n_other:,} / 同色あり {r1.n_same:,}）"
        )
        print(
            f"{'y':>3} | {'1枚以上':>8} {'対照':>8} {'lift':>8}"
            f" | {'対子':>8} {'対照':>8} {'lift':>8}"
            f" | {'[1枚:他色]':>11}{'[1枚:同色]':>11}"
        )
        for y in range(1, 10):
            print(
                f"{y:>3} | {r1.cond[y - 1]:>8.3f} {r1.control[y - 1]:>8.3f} "
                f"{r1.cond[y - 1] - r1.control[y - 1]:>+8.3f}"
                f" | {r2.cond[y - 1]:>8.3f} {r2.control[y - 1]:>8.3f} "
                f"{r2.cond[y - 1] - r2.control[y - 1]:>+8.3f}"
                f" | {r1.other[y - 1]:>11.3f}{r1.same[y - 1]:>11.3f}"
            )
            rows.append(
                [
                    cls,
                    y,
                    r1.cond[y - 1],
                    r1.control[y - 1],
                    r1.cond[y - 1] - r1.control[y - 1],
                    r2.cond[y - 1],
                    r2.control[y - 1],
                    r2.cond[y - 1] - r2.control[y - 1],
                    r1.other[y - 1],
                    r1.same[y - 1],
                    float(r1.n_cond),
                    float(r1.n_ctrl),
                ]
            )

    for min_count, summ in ((1, s1), (2, s2)):
        kind = "1 枚以上" if min_count == 1 else "対子（2 枚以上）"
        print(f"\n\n# 切られている内側牌のうち最も中心寄りのクラス別  [{kind}]")
        print("# 逆切り群（最内クラス >= 2）の巡目分布で標準化した所持率")
        for ci, cls in enumerate(CLASSES):
            label = "1/9" if cls == 1 else "2/8"
            weight = summ.inner_n[ci, 2:].sum(axis=0).astype(np.float64)
            print(f"\n## {label} の手出し")
            print(
                f"{'最内':>5} {'n':>11}   "
                + "  ".join(f"{'y' + str(y):>6}" for y in range(1, 10))
            )
            for ibin in range(MAX_INNER):
                n_j = summ.inner_n[ci, ibin]
                n = int(n_j.sum())
                if n < 500:
                    continue
                name = "なし" if ibin == 0 else f"{ibin}"
                std = standardized_rate(n_j, summ.inner_sum[ci, ibin], weight)
                print(f"{name:>5} {n:>11,d}   " + "  ".join(f"{r:>6.3f}" for r in std))

    report_window(part, menzen_only)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_csv,
            np.array(rows, dtype=np.float64),
            delimiter=",",
            fmt=["%d", "%d"] + ["%.4f"] * 8 + ["%d", "%d"],
            header=(
                "cls,y,cond1,control1,lift1,cond2,control2,lift2,"
                "other1,same1,cond_n,ctrl_n"
            ),
            comments="",
        )
        print(f"\n[csv] {out_csv}")


def save_cache(part: Partial, path: Path) -> None:
    """集計カウンタを保存する（図だけ作り直せるようにするため）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=np.asarray(part.counts, dtype=np.int64),
        counts2=np.asarray(part.counts2, dtype=np.int64),
        w1_counts=np.asarray(part.w1_counts, dtype=np.int64),
        w2_counts=np.asarray(part.w2_counts, dtype=np.int64),
        inner_w2_counts=np.asarray(part.inner_w2_counts, dtype=np.int64),
        inner_counts=np.asarray(part.inner_counts, dtype=np.int64),
        inner_counts2=np.asarray(part.inner_counts2, dtype=np.int64),
        meta=np.array([part.triggers, part.observations, part.bad_handsize], dtype=np.int64),
    )


def load_cache(path: Path) -> Partial:
    """保存した集計カウンタを読み戻す."""
    with np.load(path) as d:
        part = Partial()
        part.counts = d["counts"].tolist()
        part.counts2 = d["counts2"].tolist()
        part.w1_counts = d["w1_counts"].tolist()
        part.w2_counts = d["w2_counts"].tolist()
        part.inner_w2_counts = d["inner_w2_counts"].tolist()
        part.inner_counts = d["inner_counts"].tolist()
        part.inner_counts2 = d["inner_counts2"].tolist()
        triggers, observations, bad = (int(v) for v in d["meta"])
    part.triggers = triggers
    part.observations = observations
    part.bad_handsize = bad
    return part


def run(
    data_dir: Path | None,
    limit: int,
    no_reach: bool,
    workers: int,
    out_csv: Path | None,
    cache: Path | None,
    from_cache: bool,
) -> None:
    """データ一式を並列に集計して結果を出力する."""
    if from_cache:
        if cache is None:
            raise SystemExit("--from-cache には --cache が要る")
        total = load_cache(cache)
        report(total, True, 0, out_csv)
        print("\n" + "=" * 72)
        report(total, False, 0, None)
        return
    if data_dir is None:
        raise SystemExit("--data-dir が要る")
    files = sorted(data_dir.glob("*.mjson"))
    if limit:
        files = files[:limit]
    chunk = max(1, len(files) // (workers * 8))
    tasks = [
        ([str(p) for p in files[i : i + chunk]], no_reach) for i in range(0, len(files), chunk)
    ]
    total = Partial()
    done = 0
    with mp.Pool(workers) as pool:
        for part in pool.imap_unordered(aggregate_chunk, tasks):
            total.merge(part)
            done += 1
            if done % 20 == 0:
                print(f"... {done}/{len(tasks)} chunks", flush=True)
    if cache is not None:
        save_cache(total, cache)
        print(f"[cache] {cache}")
    report(total, True, len(files), out_csv)
    print("\n" + "=" * 72)
    report(total, False, len(files), None)
    for name, msg in total.errors[:10]:
        print(f"[error] {name}: {msg}")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む."""
    p = argparse.ArgumentParser(description="逆切りと手出し直後の所持牌")
    p.add_argument("--data-dir", type=Path, help="*.mjson を含むディレクトリ")
    p.add_argument("--limit", type=int, default=0, help="先頭 N ファイルのみ（0 で全件）")
    p.add_argument("--no-reach", action="store_true", help="リーチ受理済みの観測を除外する")
    p.add_argument("--workers", type=int, default=8, help="並列プロセス数")
    p.add_argument("--out-csv", type=Path, default=None, help="結果 CSV の出力先")
    p.add_argument("--cache", type=Path, default=None, help="集計カウンタの保存先 (.npz)")
    p.add_argument(
        "--from-cache", action="store_true", help="牌譜を読まず --cache から表だけ作る"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.data_dir,
        args.limit,
        args.no_reach,
        args.workers,
        args.out_csv,
        args.cache,
        args.from_cache,
    )
