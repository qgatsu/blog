"""間隔切り（牌のペアを間隔をあけて切る）と、2枚目の手出し直後の所持牌を集計する.

パターン（同一色 s 内）
    プレイヤー P が (s, a) を捨て（手出し・ツモ切りを問わない）、間隔をあけて
    (s, b) を手出しした。間隔は打牌 index で j >= i + 2（直後 j = i + 1 は含めない）。
ペア
    canonical 低位側で (a, b) = (1,2), (2,4), (3,5), (4,6)。
    鏡像 (9,8), (8,6), (7,5), (6,4) は canonical y = 10 - y に写して統合する。
測定
    b の手出し直後の同色門前手牌に canonical 数 y = 1..9 があるか（0/1）。
    注目するのは y = a + 3（12 → 4, 24 → 5, 35 → 6, 46 → 7）。
対照
    同じ b の手出しのうち、資格のある先行 a が無いもの（base）。
    打牌 index 別に集計し、cond 群の index 分布で標準化してから比べる。
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
SUITS: str = "mps"
SUIT_OFFSET: dict[str, int] = {"m": 0, "p": 9, "s": 18}

# canonical 低位側のペア (a, b)。注目する所持牌は a + 3。
PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (2, 4), (3, 5), (4, 6))
N_PAIR: int = len(PAIRS)
MAX_J: int = 20  # 打牌 index のビン数（これ以上は最終ビンに寄せる）
MAX_LAG: int = 12  # 先行 a との間隔 (j - i) のビン数
N_MASK: int = 512  # 同色 9 牌の所持を表すビットマスク
MIN_GAP: int = 2  # 「間隔をあけた」とみなす最小の j - i
# 2 つ目の手出しを、先行する 1 つ目の有無・間隔・間に挟んだ字牌で分ける
STATE_NONE: int = 0  # 先行なし
STATE_ADJACENT: int = 1  # 間隔なし（直前に切っただけ）
STATE_PLAIN: int = 2  # 間隔あり・間に字牌なし
STATE_HONOR_TSUMOGIRI: int = 3  # 間隔あり・間に字牌のツモ切りのみ
STATE_HONOR_TEDASHI: int = 4  # 間隔あり・間に字牌の手出し
N_STATE: int = 5
GAP_STATES: tuple[int, int, int] = (STATE_PLAIN, STATE_HONOR_TSUMOGIRI, STATE_HONOR_TEDASHI)
N_GAP_STATE: int = len(GAP_STATES)
# 手出しとツモ切りで所持率に差が出なかったため、出力では字牌の有無だけで分ける。
# 集計は内訳を保ったままにしてあるので、必要なら分けて見られる。
GAP_HONOR_NONE: tuple[int] = (0,)
GAP_HONOR_ANY: tuple[int, int] = (1, 2)
STATE_LABEL: tuple[str, ...] = (
    "先行なし",
    "間隔なし",
    "字牌なし",
    "字牌ツモ切り",
    "字牌手出し",
)

# 手出しされた牌の数 -> [(ペア番号, 鏡像か, 先行牌 a の数), ...]
B_MAP: dict[int, tuple[tuple[int, bool, int], ...]] = {}
for _pi, (_a, _b) in enumerate(PAIRS):
    B_MAP.setdefault(_b, ())
    B_MAP[_b] += ((_pi, False, _a),)
    B_MAP.setdefault(10 - _b, ())
    B_MAP[10 - _b] += ((_pi, True, 10 - _a),)


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
        default_factory=lambda: [0] * (N_PAIR * 2 * MAX_J * N_STATE * N_MASK)
    )
    gap_counts: list[int] = field(
        default_factory=lambda: [0] * (N_PAIR * 2 * MAX_LAG * N_GAP_STATE * N_MASK)
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
        la, lb = self.gap_counts, other.gap_counts
        for i, v in enumerate(lb):
            if v:
                la[i] += v
        self.triggers += other.triggers
        self.observations += other.observations
        self.bad_handsize += other.bad_handsize
        self.errors.extend(other.errors)


def aggregate_game(events: list[dict[str, object]], part: Partial, no_reach: bool) -> None:
    """1 半荘を頭から再生し、間隔切りの成立/非成立ごとの所持を部分集計に加える."""
    hands: list[list[int]] = []
    melds: list[int] = []
    dcount: list[int] = []
    # discards[actor][tile_index] = その牌を捨てた打牌 index（0 始まり）の履歴
    discards: list[list[list[int]]] = []
    # 字牌を切った打牌 index（手出し / ツモ切り 別）
    honor_tedashi: list[list[int]] = []
    honor_tsumogiri: list[list[int]] = []
    reach_active = False
    counts = part.counts
    gap_counts = part.gap_counts

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
            discards = [[[] for _ in range(34)] for _ in range(4)]
            honor_tedashi = [[] for _ in range(4)]
            honor_tsumogiri = [[] for _ in range(4)]
            reach_active = False
            continue
        if not hands:
            continue
        if etype == "tsumo":
            hands[cast(int, ev["actor"])][tile_index(canon(cast(str, ev["pai"])))] += 1
        elif etype == "dahai":
            actor = cast(int, ev["actor"])
            pai = canon(cast(str, ev["pai"]))
            idx = tile_index(pai)
            hand = hands[actor]
            hand[idx] -= 1
            j = dcount[actor]
            dcount[actor] += 1
            if sum(hand) != 13 - 3 * melds[actor]:
                part.bad_handsize += 1
            if pai not in HONORS:
                val = int(pai[0])
                targets = B_MAP.get(val)
                if (
                    targets is not None
                    and not cast(bool, ev["tsumogiri"])
                    and not (no_reach and reach_active)
                ):
                    suit = pai[1]
                    off = SUIT_OFFSET[suit]
                    mask = 0
                    for k in range(9):
                        if hand[off + k]:
                            mask |= 1 << k
                    rev_mask = REV_TABLE[mask]
                    fuuro = 1 if melds[actor] else 0
                    jbin = j if j < MAX_J else MAX_J - 1
                    for pair_i, is_high, a_val in targets:
                        prior = discards[actor][off + a_val - 1]
                        # 直近の先行 a を探す。間隔が MIN_GAP 以上なら「間隔あり」、
                        # 直前（間隔 1）でしか切っていなければ「間隔なし」、無ければ「先行なし」。
                        gap = -1
                        adjacent = False
                        for i in reversed(prior):
                            d = j - i
                            if d >= MIN_GAP:
                                gap = d
                                break
                            if d == 1:
                                adjacent = True
                        if gap >= 0:
                            # 1 つ目と 2 つ目の間（両端は含めない）に字牌を切っているか
                            start = j - gap
                            if any(start < k < j for k in honor_tedashi[actor]):
                                state = STATE_HONOR_TEDASHI
                            elif any(start < k < j for k in honor_tsumogiri[actor]):
                                state = STATE_HONOR_TSUMOGIRI
                            else:
                                state = STATE_PLAIN
                        elif adjacent:
                            state = STATE_ADJACENT
                        else:
                            state = STATE_NONE
                        m = rev_mask if is_high else mask
                        slot = ((pair_i * 2 + fuuro) * MAX_J + jbin) * N_STATE + state
                        counts[slot * N_MASK + m] += 1
                        part.observations += 1
                        if gap >= 0:
                            part.triggers += 1
                            gbin = gap if gap < MAX_LAG else MAX_LAG - 1
                            gslot = ((pair_i * 2 + fuuro) * MAX_LAG + gbin) * N_GAP_STATE + (
                                state - STATE_PLAIN
                            )
                            gap_counts[gslot * N_MASK + m] += 1
            else:
                if cast(bool, ev["tsumogiri"]):
                    honor_tsumogiri[actor].append(j)
                else:
                    honor_tedashi[actor].append(j)
            discards[actor][idx].append(j)
        elif etype == "reach_accepted":
            reach_active = True
        elif etype in ("pon", "chi", "daiminkan", "ankan"):
            actor = cast(int, ev["actor"])
            for t in cast("list[str]", ev["consumed"]):
                hands[actor][tile_index(canon(t))] -= 1
            melds[actor] += 1
        elif etype == "kakan":
            # consumed は既にポン済みの 3 枚。手牌から出るのは pai の 1 枚だけで、副露数も増えない。
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
    """ペアごとの集計結果（打牌 index 別・間隔別の生カウント）."""

    n: npt.NDArray[np.int64]  # (pair, j, state)
    sums: npt.NDArray[np.int64]  # (pair, j, state, y)
    gap_n: npt.NDArray[np.int64]  # (pair, gap, gap_state)
    gap_sum: npt.NDArray[np.int64]  # (pair, gap, gap_state, y)


def summarize(part: Partial, menzen_only: bool) -> Summary:
    """平坦な集計を (ペア, 打牌 index / 間隔, 状態, y) の形にほどく."""
    bits = mask_bit_table()
    # 第 2 軸は副露フラグ（0 = 門前, 1 = 副露あり）
    arr = np.asarray(part.counts, dtype=np.int64).reshape(N_PAIR, 2, MAX_J, N_STATE, N_MASK)
    gap = np.asarray(part.gap_counts, dtype=np.int64).reshape(
        N_PAIR, 2, MAX_LAG, N_GAP_STATE, N_MASK
    )
    if menzen_only:
        arr = arr[:, 0]
        gap = gap[:, 0]
    else:
        arr = arr.sum(axis=1)
        gap = gap.sum(axis=1)
    return Summary(
        n=arr.sum(axis=-1),
        sums=arr @ bits,
        gap_n=gap.sum(axis=-1),
        gap_sum=gap @ bits,
    )


def standardized_rate(
    n: npt.NDArray[np.int64],
    s: npt.NDArray[np.int64],
    weight: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """層別の所持数を、与えた層の分布で標準化した所持率に直す.

    n は層ごとの観測数、s は層ごとの所持数 (層, 9)、weight は合わせる先の分布。
    """
    nf = n.astype(np.float64)
    valid = (nf > 0) & (weight > 0)
    if not valid.any():
        return np.full(9, np.nan)
    rate = np.zeros((nf.shape[0], 9), dtype=np.float64)
    rate[valid] = s.astype(np.float64)[valid] / nf[valid][:, None]
    w = weight * valid
    return (rate * w[:, None]).sum(axis=0) / w.sum()


def report(part: Partial, menzen_only: bool, n_files: int, out_csv: Path | None) -> None:
    """集計結果を表として出力する.

    主指標は「間隔あり」対「それ以外」で、対照には直前に切っただけの場合も含める。
    そのうえで、間隔あり群を「間に挟んだ字牌」で分けて比べる（間隔の分布は揃える）。
    """
    summary = summarize(part, menzen_only)
    scope = "門前のみ" if menzen_only else "副露込み"
    print(f"# 間隔切り  files={n_files}  scope={scope}")
    print(
        f"# 観測 {part.observations:,}  間隔あり {part.triggers:,}  "
        f"handsize不整合 {part.bad_handsize:,}  errors {len(part.errors)}"
    )
    rows: list[list[float]] = []
    for pair_i, (a, b) in enumerate(PAIRS):
        n_st = summary.n[pair_i]
        s_st = summary.sums[pair_i]
        totals = n_st.sum(axis=0)
        cond_n = n_st[:, GAP_STATES].sum(axis=1)
        cond_s = s_st[:, GAP_STATES, :].sum(axis=1)
        weight = cond_n.astype(np.float64)
        ctrl_n = n_st[:, STATE_NONE] + n_st[:, STATE_ADJACENT]
        ctrl_s = s_st[:, STATE_NONE, :] + s_st[:, STATE_ADJACENT, :]
        cond = cond_s.sum(axis=0) / max(cond_n.sum(), 1)
        ctrl = standardized_rate(ctrl_n, ctrl_s, weight)
        none_rate = standardized_rate(n_st[:, STATE_NONE], s_st[:, STATE_NONE, :], weight)
        adj_rate = standardized_rate(n_st[:, STATE_ADJACENT], s_st[:, STATE_ADJACENT, :], weight)
        target = a + 3
        print(
            f"\n## {a}{b} → 注目 y={target}   間隔あり n={cond_n.sum():,}  "
            f"対照 n={ctrl_n.sum():,}（うち間隔なし {totals[STATE_ADJACENT]:,}）"
        )
        print(
            f"{'y':>3} {'間隔あり':>9} {'対照':>8} {'lift':>8}"
            f"{'[先行なし]':>12}{'[間隔なし]':>12}"
        )
        for y in range(1, 10):
            mark = "  ← 注目" if y == target else ""
            print(
                f"{y:>3} {cond[y - 1]:>9.3f} {ctrl[y - 1]:>8.3f} "
                f"{cond[y - 1] - ctrl[y - 1]:>+8.3f} "
                f"{none_rate[y - 1]:>11.3f} {adj_rate[y - 1]:>11.3f}{mark}"
            )
            rows.append(
                [
                    a,
                    b,
                    y,
                    cond[y - 1],
                    ctrl[y - 1],
                    cond[y - 1] - ctrl[y - 1],
                    none_rate[y - 1],
                    adj_rate[y - 1],
                    float(cond_n.sum()),
                    float(ctrl_n.sum()),
                ]
            )

    print("\n\n# 間に字牌を挟んだか（間隔の分布を間隔あり全体に揃えた比較）")
    print(
        f"\n{'ペア':>6} {'注目':>4} {'字牌なし':>10} {'字牌あり':>10} {'差':>9}"
        f"   {'n(なし)':>10} {'n(あり)':>10}"
        f"   {'[内訳] ツモ切り':>16} {'手出し':>9}"
    )
    for pair_i, (a, b) in enumerate(PAIRS):
        gn = summary.gap_n[pair_i]  # (gap, gap_state)
        gs = summary.gap_sum[pair_i]  # (gap, gap_state, y)
        weight = gn.sum(axis=1).astype(np.float64)
        target = a + 3
        groups: list[float] = []
        for idx in (GAP_HONOR_NONE, GAP_HONOR_ANY):
            rate = standardized_rate(
                gn[:, idx].sum(axis=1), gs[:, idx, :].sum(axis=1), weight
            )
            groups.append(float(rate[target - 1]))
        detail: list[float] = []
        for k in (1, 2):
            rate = standardized_rate(gn[:, k], gs[:, k, :], weight)
            detail.append(float(rate[target - 1]))
        print(
            f"{a}{b:<5} {target:>4} {groups[0]:>10.3f} {groups[1]:>10.3f} "
            f"{groups[1] - groups[0]:>+9.3f}   "
            f"{gn[:, GAP_HONOR_NONE].sum():>10,d} {gn[:, GAP_HONOR_ANY].sum():>10,d}"
            f"   {detail[0]:>16.3f} {detail[1]:>9.3f}"
        )

    print("\n\n# 間隔の長さ別（注目牌の所持率）  全体 / 字牌なし / 字牌あり")
    for pair_i, (a, _b) in enumerate(PAIRS):
        gn = summary.gap_n[pair_i]
        gs = summary.gap_sum[pair_i]
        target = a + 3
        print(f"\n## {a}{_b} → {target}")
        print(f"{'gap':>4} {'全体':>8} {'n':>9}   {'字牌なし':>9} {'n':>9}   {'字牌あり':>9} {'n':>9}")
        for g in range(MIN_GAP, MAX_LAG):
            tot_n = int(gn[g].sum())
            if tot_n < 200:
                continue
            tot_r = gs[g, :, target - 1].sum() / tot_n
            cells = f"{g if g < MAX_LAG - 1 else str(g) + '+':>4} {tot_r:>8.3f} {tot_n:>9,d}   "
            for idx in (GAP_HONOR_NONE, GAP_HONOR_ANY):
                n_k = int(gn[g, idx].sum())
                if n_k < 200:
                    cells += f"{'-':>9} {n_k:>9,d}   "
                else:
                    hit = float(gs[g, idx, target - 1].sum())
                    cells += f"{hit / n_k:>9.3f} {n_k:>9,d}   "
            print(cells)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_csv,
            np.array(rows, dtype=np.float64),
            delimiter=",",
            fmt=["%d", "%d", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%d", "%d"],
            header="a,b,y,cond,control,lift,ctrl_none,ctrl_adjacent,cond_n,ctrl_n",
            comments="",
        )
        print(f"\n[csv] {out_csv}")


def save_cache(part: Partial, path: Path) -> None:
    """集計カウンタを保存する（図だけ作り直せるようにするため）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=np.asarray(part.counts, dtype=np.int64),
        gap_counts=np.asarray(part.gap_counts, dtype=np.int64),
        meta=np.array(
            [part.triggers, part.observations, part.bad_handsize], dtype=np.int64
        ),
    )


def load_cache(path: Path) -> Partial:
    """保存した集計カウンタを読み戻す."""
    with np.load(path) as d:
        part = Partial()
        part.counts = d["counts"].tolist()
        part.gap_counts = d["gap_counts"].tolist()
        triggers, observations, bad = (int(v) for v in d["meta"])
    part.triggers = triggers
    part.observations = observations
    part.bad_handsize = bad
    return part


def run(
    data_dir: Path,
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
    p = argparse.ArgumentParser(description="間隔切りと2枚目の手出し直後の所持牌")
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
