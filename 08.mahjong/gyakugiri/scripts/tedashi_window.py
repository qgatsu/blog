"""数牌の打牌の周りに牌があるかを、位置ごと・巡目ごとに測る.

先行条件は付けない。「その牌をその巡目に切った」ことだけを条件にする。

3 状態（同じ打牌時点の同じ位置を、起きたことで分ける）
    手出し      … その打牌でその位置が手出しされた。
    ツモ切り    … その打牌でその位置がツモ切りされた。
    切っていない … その打牌でその位置には何も起きていない（3 色 × 位置 1..9 のうち、
                   実際に切られた位置を除いたすべて）。ツモ切り自体が負の情報を持つため、
                   base はここに取る。
測定
    切った（あるいは何も起きていない）位置から見て、同じ色の 1 つ外側 / 2 つ外側 /
    1 つ内側 / 2 つ内側 の牌を 1 枚以上持っているか。内側は中心（5）に向かう向き。
    位置ごとに別々に出し、そのクラスで存在する位置の平均も出す。
    端に寄った牌ほど外側が盤外になるため、平均は存在する位置だけで取る。
集計の作り
    打牌のたびに 3 色それぞれの「同色 9 牌の所持マスク」を数えておき、
    切られた位置の分を引いて「切っていない」を作る。
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

MAX_J: int = 20
N_MASK: int = 512
N_STATE: int = 2  # 0 = 手出し, 1 = ツモ切り
CLASS_LABEL: tuple[str, str, str, str, str] = ("1/9", "2/8", "3/7", "4/6", "5")
# canonical 座標でのオフセット（+ が中心方向）
OFFSETS: tuple[int, int, int, int] = (-2, -1, 1, 2)
OFFSET_LABEL: tuple[str, str, str, str] = ("2つ外側", "1つ外側", "1つ内側", "2つ内側")


def canon(tile: str) -> str:
    """赤5（末尾 r）を通常牌に正規化する."""
    return tile[:2] if (len(tile) == 3 and tile[2] == "r") else tile


def tile_index(tile: str) -> int:
    """正規化済みの牌を 0..33 の索引に写す."""
    if tile in HONORS:
        return 27 + "ESWNPFC".index(tile)
    return SUIT_OFFSET[tile[1]] + int(tile[0]) - 1


_MELD_CACHE: dict[tuple[int, ...], int] = {}
_FREE_CACHE: dict[tuple[int, ...], int] = {}


def _max_melds(c: list[int], i: int) -> int:
    """位置 i 以降で作れる面子（順子・刻子）の最大数を数える."""
    while i < 9 and c[i] == 0:
        i += 1
    if i >= 9:
        return 0
    best = 0
    if c[i] >= 3:
        c[i] -= 3
        best = max(best, 1 + _max_melds(c, i))
        c[i] += 3
    if i + 2 < 9 and c[i + 1] and c[i + 2]:
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        best = max(best, 1 + _max_melds(c, i))
        c[i] += 1
        c[i + 1] += 1
        c[i + 2] += 1
    # この 1 枚を面子に使わず余らせる
    c[i] -= 1
    best = max(best, _max_melds(c, i))
    c[i] += 1
    return best


def max_melds(counts: tuple[int, ...]) -> int:
    """同色 9 位置の枚数から、作れる面子の最大数を返す."""
    v = _MELD_CACHE.get(counts)
    if v is None:
        v = _max_melds(list(counts), 0)
        _MELD_CACHE[counts] = v
    return v


def free_mask(counts: tuple[int, ...]) -> int:
    """完成面子に必須でない牌だけを立てたマスクを返す.

    位置 k の牌を 1 枚抜いても最大面子数が変わらなければ、その牌は完成面子に
    必須ではない（未完成のブロックか浮き牌）とみなして立てる。
    567 を持っているときの 7 は抜くと面子数が減るので落ちる。
    77 のような対子は抜いても面子数が変わらないので残る。
    """
    v = _FREE_CACHE.get(counts)
    if v is not None:
        return v
    base_melds = max_melds(counts)
    mask = 0
    work = list(counts)
    for k in range(9):
        if work[k]:
            work[k] -= 1
            if max_melds(tuple(work)) == base_melds:
                mask |= 1 << k
            work[k] += 1
    _FREE_CACHE[counts] = mask
    return mask


@dataclass
class Partial:
    """部分集計."""

    # 全打牌 × 3 色の所持マスク（巡目別）
    suit_counts: list[int] = field(default_factory=lambda: [0] * (MAX_J * N_MASK))
    # 位置 v (1..9) が切られたときの所持マスク（巡目・手出しかツモ切りか別）
    cut_counts: list[int] = field(
        default_factory=lambda: [0] * (9 * MAX_J * N_STATE * N_MASK)
    )
    # 完成面子に必須な牌を落としたマスクでの同じ集計
    suit_free: list[int] = field(default_factory=lambda: [0] * (MAX_J * N_MASK))
    cut_free: list[int] = field(
        default_factory=lambda: [0] * (9 * MAX_J * N_STATE * N_MASK)
    )
    bad_handsize: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: Partial) -> None:
        """別の部分集計を足し込む."""
        for mine, theirs in (
            (self.suit_counts, other.suit_counts),
            (self.cut_counts, other.cut_counts),
            (self.suit_free, other.suit_free),
            (self.cut_free, other.cut_free),
        ):
            for i, v in enumerate(theirs):
                if v:
                    mine[i] += v
        self.bad_handsize += other.bad_handsize
        self.errors.extend(other.errors)


def aggregate_game(events: list[dict[str, object]], part: Partial, no_reach: bool) -> None:
    """1 半荘を再生し、打牌ごとに 3 色の所持マスクを部分集計に加える."""
    hands: list[list[int]] = []
    melds: list[int] = []
    dcount: list[int] = []
    reach_active = False
    suit_counts = part.suit_counts
    cut_counts = part.cut_counts
    suit_free = part.suit_free
    cut_free = part.cut_free

    for ev in events:
        etype = cast(str, ev["type"])
        if etype == "start_kyoku":
            hands = [[0] * 34 for _ in range(4)]
            for actor, th in enumerate(cast("list[list[str]]", ev["tehais"])):
                for t in th:
                    hands[actor][tile_index(canon(t))] += 1
            melds = [0, 0, 0, 0]
            dcount = [0, 0, 0, 0]
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
            # 門前・リーチ前のみを対象にする
            if melds[actor] or (no_reach and reach_active):
                continue
            jbin = j if j < MAX_J else MAX_J - 1
            base = jbin * N_MASK
            masks: list[int] = []
            frees: list[int] = []
            for off in (0, 9, 18):
                col = tuple(hand[off : off + 9])
                m = 0
                for k in range(9):
                    if col[k]:
                        m |= 1 << k
                f = free_mask(col)
                masks.append(m)
                frees.append(f)
                suit_counts[base + m] += 1
                suit_free[base + f] += 1
            if pai not in HONORS:
                val = int(pai[0])
                si = SUIT_OFFSET[pai[1]] // 9
                state = 1 if cast(bool, ev["tsumogiri"]) else 0
                slot = ((val - 1) * MAX_J + jbin) * N_STATE + state
                cut_counts[slot * N_MASK + masks[si]] += 1
                cut_free[slot * N_MASK + frees[si]] += 1
        elif etype == "reach_accepted":
            reach_active = True
        elif etype in ("pon", "chi", "daiminkan", "ankan"):
            actor = cast(int, ev["actor"])
            for t in cast("list[str]", ev["consumed"]):
                hands[actor][tile_index(canon(t))] -= 1
            melds[actor] += 1
        elif etype == "kakan":
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
    """(マスク, 位置) -> その位置を 1 枚以上持っていれば 1."""
    table = np.zeros((N_MASK, 9), dtype=np.int64)
    for m in range(N_MASK):
        for k in range(9):
            if m & (1 << k):
                table[m, k] = 1
    return table


@dataclass(frozen=True)
class Rates:
    """位置 v・巡目 j・状態ごとの観測数と所持数."""

    n: npt.NDArray[np.float64]  # (v, j, state) state: 0=手出し, 1=ツモ切り, 2=切っていない
    hit: npt.NDArray[np.float64]  # (v, j, state, u) u = 見る位置 0..8


def compute(part: Partial, free_only: bool = False) -> Rates:
    """集計をほどき、「切っていない」を全体から引いて作る.

    free_only が真なら、完成面子に必須な牌を落としたマスクのほうを使う。
    """
    bits = mask_bit_table()
    src_suit = part.suit_free if free_only else part.suit_counts
    src_cut = part.cut_free if free_only else part.cut_counts
    suit = np.asarray(src_suit, dtype=np.int64).reshape(MAX_J, N_MASK)
    cut = np.asarray(src_cut, dtype=np.int64).reshape(9, MAX_J, N_STATE, N_MASK)
    total_n = suit.sum(axis=-1).astype(np.float64)  # (j,)
    total_hit = (suit @ bits).astype(np.float64)  # (j, u)
    cut_n = cut.sum(axis=-1).astype(np.float64)  # (v, j, state)
    cut_hit = (cut @ bits).astype(np.float64)  # (v, j, state, u)
    none_n = total_n[None, :] - cut_n.sum(axis=-1)  # (v, j)
    none_hit = total_hit[None, :, :] - cut_hit.sum(axis=-2)  # (v, j, u)
    n = np.concatenate([cut_n, none_n[:, :, None]], axis=-1)
    hit = np.concatenate([cut_hit, none_hit[:, :, None, :]], axis=-2)
    return Rates(n=n, hit=hit)


def offset_positions(v: int, d: int) -> int | None:
    """位置 v（1..9）から canonical のオフセット d だけ動いた位置。盤外なら None."""
    step = d if v <= 5 else -d  # 高位側は鏡なので向きを反転する
    u = v + step
    return u if 1 <= u <= 9 else None


def rate_of(rates: Rates, v: int, j: int, state: int, u: int) -> tuple[float, float]:
    """位置 v をその状態で扱ったときの、位置 u の所持数と観測数を返す."""
    return float(rates.hit[v - 1, j, state, u - 1]), float(rates.n[v - 1, j, state])


def report_mid(rates: Rates, min_n: int, title: str) -> None:
    """中盤（巡目 7-12）に絞ったまとめを出す."""
    print(f"\n\n# 中盤（巡目 7-12）まとめ  [{title}]")
    print("# 各セル: 手出し / ツモ切り / 切っていない。差は base の水準で圧縮されるので OR も出す")
    print(
        f"{'クラス':>6} {'位置':>8} {'手出し':>8} {'ツモ切り':>9} {'base':>8} "
        f"{'lift':>8} {'OR(手出)':>9} {'OR(ツモ)':>9} {'n(手出)':>12}"
    )
    mid = range(6, 12)
    for ci in range(5):
        cls = ci + 1
        members = [cls] if cls == 5 else [cls, 10 - cls]
        for di, d in enumerate(OFFSETS):
            hit = [0.0, 0.0, 0.0]
            obs = [0.0, 0.0, 0.0]
            ok = True
            for v in members:
                u = offset_positions(v, d)
                if u is None:
                    ok = False
                    break
                for st in range(3):
                    for j in mid:
                        h, n = rate_of(rates, v, j, st, u)
                        hit[st] += h
                        obs[st] += n
            if not ok or obs[0] < min_n:
                continue
            r = [hit[st] / obs[st] for st in range(3)]
            print(
                f"{CLASS_LABEL[ci]:>6} {OFFSET_LABEL[di]:>8} {r[0]:>8.3f} {r[1]:>9.3f} "
                f"{r[2]:>8.3f} {r[0] - r[2]:>+8.3f} {_odds(r[0], r[2]):>9.2f} "
                f"{_odds(r[1], r[2]):>9.2f} {obs[0]:>12,.0f}"
            )


def _odds(a: float, b: float) -> float:
    """オッズ比を返す（どちらかが 0 か 1 なら nan）."""
    return (a / (1 - a)) / (b / (1 - b)) if 0 < a < 1 and 0 < b < 1 else float("nan")


def report(part: Partial, n_files: int, min_n: int) -> None:
    """位置ごと・巡目ごとの所持率を出す."""
    rates = compute(part)
    print(f"# 打牌の周りの所持  files={n_files}  門前・リーチ前")
    print(f"# handsize不整合 {part.bad_handsize:,}  errors {len(part.errors)}")
    print("# 各セル: 手出し / ツモ切り / 切っていない")

    for ci in range(5):
        cls = ci + 1
        # そのクラスに属する実際の位置（1/9 なら 1 と 9）
        members = [cls] if cls == 5 else [cls, 10 - cls]
        print(f"\n\n## {CLASS_LABEL[ci]} を切ったとき")
        cols = [lab for lab in OFFSET_LABEL]
        print(f"{'巡目':>4}  " + "  ".join(f"{c:>22}" for c in cols) + f"  {'平均lift':>9}")
        for j in range(MAX_J):
            cells: list[str] = []
            lifts: list[float] = []
            shown = False
            for d in OFFSETS:
                hit = [0.0, 0.0, 0.0]
                obs = [0.0, 0.0, 0.0]
                ok = True
                for v in members:
                    u = offset_positions(v, d)
                    if u is None:
                        ok = False
                        break
                    for st in range(3):
                        h, n = rate_of(rates, v, j, st, u)
                        hit[st] += h
                        obs[st] += n
                if not ok or min(obs[0], obs[1]) < min_n:
                    cells.append(f"{'-':>22}")
                    continue
                shown = True
                r = [hit[st] / obs[st] if obs[st] else float("nan") for st in range(3)]
                cells.append(f"{r[0]:>6.3f}/{r[1]:>6.3f}/{r[2]:>6.3f}")
                lifts.append(r[0] - r[2])
            if shown:
                turn = f"{j + 1}" if j < MAX_J - 1 else f"{j + 1}+"
                avg = f"{sum(lifts) / len(lifts):>+9.3f}" if lifts else f"{'-':>9}"
                print(f"{turn:>4}  " + "  ".join(cells) + f"  {avg}")

    print("\n\n# 手出しの平均リフト（切っていない を base に、窓内の位置で平均）")
    head = "  ".join(f"{lab:>8}" for lab in CLASS_LABEL)
    print(f"{'巡目':>4}  {head}")
    for j in range(MAX_J):
        cells = []
        shown = False
        for ci in range(5):
            cls = ci + 1
            members = [cls] if cls == 5 else [cls, 10 - cls]
            lifts: list[float] = []
            for d in OFFSETS:
                hit = [0.0, 0.0]
                obs = [0.0, 0.0]
                ok = True
                for v in members:
                    u = offset_positions(v, d)
                    if u is None:
                        ok = False
                        break
                    for idx, st in enumerate((0, 2)):
                        h, n = rate_of(rates, v, j, st, u)
                        hit[idx] += h
                        obs[idx] += n
                if not ok or obs[0] < min_n:
                    continue
                lifts.append(hit[0] / obs[0] - hit[1] / obs[1])
            if not lifts:
                cells.append(f"{'-':>8}")
                continue
            shown = True
            cells.append(f"{sum(lifts) / len(lifts):>+8.3f}")
        if shown:
            turn = f"{j + 1}" if j < MAX_J - 1 else f"{j + 1}+"
            print(f"{turn:>4}  " + "  ".join(cells))

    report_mid(rates, min_n, "すべての所持")
    report_mid(compute(part, free_only=True), min_n, "完成面子に必須な牌を除く")

    print("\n\n# ツモ切りの平均リフト（同じく 切っていない を base に）")
    print(f"{'巡目':>4}  {head}")
    for j in range(MAX_J):
        cells = []
        shown = False
        for ci in range(5):
            cls = ci + 1
            members = [cls] if cls == 5 else [cls, 10 - cls]
            lifts = []
            for d in OFFSETS:
                hit = [0.0, 0.0]
                obs = [0.0, 0.0]
                ok = True
                for v in members:
                    u = offset_positions(v, d)
                    if u is None:
                        ok = False
                        break
                    for idx, st in enumerate((1, 2)):
                        h, n = rate_of(rates, v, j, st, u)
                        hit[idx] += h
                        obs[idx] += n
                if not ok or obs[0] < min_n:
                    continue
                lifts.append(hit[0] / obs[0] - hit[1] / obs[1])
            if not lifts:
                cells.append(f"{'-':>8}")
                continue
            shown = True
            cells.append(f"{sum(lifts) / len(lifts):>+8.3f}")
        if shown:
            turn = f"{j + 1}" if j < MAX_J - 1 else f"{j + 1}+"
            print(f"{turn:>4}  " + "  ".join(cells))


def save_cache(part: Partial, path: Path) -> None:
    """集計カウンタを保存する."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        suit_counts=np.asarray(part.suit_counts, dtype=np.int64),
        cut_counts=np.asarray(part.cut_counts, dtype=np.int64),
        suit_free=np.asarray(part.suit_free, dtype=np.int64),
        cut_free=np.asarray(part.cut_free, dtype=np.int64),
        meta=np.array([part.bad_handsize], dtype=np.int64),
    )


def load_cache(path: Path) -> Partial:
    """保存した集計カウンタを読み戻す."""
    with np.load(path) as d:
        part = Partial()
        part.suit_counts = d["suit_counts"].tolist()
        part.cut_counts = d["cut_counts"].tolist()
        part.suit_free = d["suit_free"].tolist()
        part.cut_free = d["cut_free"].tolist()
        bad = int(d["meta"][0])
    part.bad_handsize = bad
    return part


def run(
    data_dir: Path | None,
    limit: int,
    no_reach: bool,
    workers: int,
    cache: Path | None,
    from_cache: bool,
    min_n: int,
) -> None:
    """データ一式を並列に集計して結果を出力する."""
    if from_cache:
        if cache is None:
            raise SystemExit("--from-cache には --cache が要る")
        report(load_cache(cache), 0, min_n)
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
    report(total, len(files), min_n)
    for name, msg in total.errors[:10]:
        print(f"[error] {name}: {msg}")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む."""
    p = argparse.ArgumentParser(description="打牌の周りの所持（位置ごと・巡目ごと）")
    p.add_argument("--data-dir", type=Path, help="*.mjson を含むディレクトリ")
    p.add_argument("--limit", type=int, default=0, help="先頭 N ファイルのみ（0 で全件）")
    p.add_argument("--no-reach", action="store_true", help="リーチ受理済みの観測を除外する")
    p.add_argument("--workers", type=int, default=8, help="並列プロセス数")
    p.add_argument("--cache", type=Path, default=None, help="集計カウンタの保存先 (.npz)")
    p.add_argument("--from-cache", action="store_true", help="牌譜を読まず --cache から表を作る")
    p.add_argument("--min-n", type=int, default=5000, help="この件数未満は伏せる")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.data_dir,
        args.limit,
        args.no_reach,
        args.workers,
        args.cache,
        args.from_cache,
        args.min_n,
    )
