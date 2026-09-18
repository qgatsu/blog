"""序盤の切りのそばの牌を持っていないか（切った牌クラス × 持っている牌クラスの 5×5）。

セオリー「序盤に切った牌のそばは持たれていない」を、切った牌と見る牌をそれぞれ端距離クラス
（1/9, 2/8, 3/7, 4/6, 5）に畳んだ 5×5 で見る。

対象
    序盤（打牌 index j ≤ CUTOFF、既定 4）の数牌の切り（手出し・ツモ切りの両方）。
軸
    行 = 切った牌のクラス cx = min(raw, 10-raw)。
    列 = 見る牌のクラス cy。見るのは「切った牌と同じ半分（中心5まで）」の cy クラス牌のみ。
    例: 1（低位）を切ったら 1,2,3,4,5 を見る（6,7,8,9=8側は見ない）。8を切ったら 5,6,7,8,9。
測定（切った直後の同色手牌）
    同じ半分にある cy クラス牌を 1 枚以上持つか（所持 P≥1）。対角 cx=cy は切った牌自身（対子）。
2指標
    指標1 単純所持率 = P(cy を所持 | 序盤に cx を切った)。
    指標2 リフト = 単純所持率 − 構造baseline。
    構造baseline = 全切りの手牌における、同絶対位置（cy と 10-cy の鏡平均）の P≥1 マージナル。
    切った牌の巡目分布で標準化し、位置ごとの「本来の持たれやすさ」を除く。

序盤 cutoff は既定 4。図は j≤4 の 2 枚（単純所持率 / リフト）。表は 4,5,6 を併記。
リーチ除外（--no-reach）。赤5は5統合、門前手牌のみ。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm

plt.switch_backend("Agg")

HONORS: frozenset[str] = frozenset("ESWNPFC")
N_J: int = 15
N_CLASS: int = 5
CLASS_LABEL: list[str] = ["1/9", "2/8", "3/7", "4/6", "5"]
CUTOFF: int = 6              # 図の序盤定義
CUTOFFS_TABLE: tuple[int, ...] = (4, 5, 6)  # 表に併記する cutoff
_JP_FONT: Path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def setup_japanese_font() -> None:
    """日本語フォント（Noto Sans CJK JP）が在れば matplotlib に登録する。"""
    if _JP_FONT.exists():
        fm.fontManager.addfont(str(_JP_FONT))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def canon(tile: str) -> str:
    """赤5（末尾 r）を通常牌に正規化する。"""
    return tile[:2] if (len(tile) == 3 and tile[2] == "r") else tile


def parse_num(tile: str) -> tuple[str, int] | None:
    """数牌なら (色, 数)、字牌なら None。"""
    return None if tile in HONORS else (tile[1], int(tile[0]))


@dataclass
class Counters:
    """(巡目, 切りクラス, 見るクラス) の切り数と所持数、および位置別マージナル。"""

    a_n: np.ndarray = field(
        default_factory=lambda: np.zeros((N_J, N_CLASS), dtype=np.int64))
    a_s: np.ndarray = field(
        default_factory=lambda: np.zeros((N_J, N_CLASS, N_CLASS), dtype=np.float64))
    m_c: np.ndarray = field(default_factory=lambda: np.zeros(N_J, dtype=np.int64))
    m_s: np.ndarray = field(default_factory=lambda: np.zeros((N_J, 10), dtype=np.int64))
    events: int = 0
    bad_handsize: int = 0


def load_events(path: Path) -> list[dict[str, object]]:
    """1 半荘の MJAI（NDJSON）を読む（I/O）。"""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_cache(path: Path, c: Counters, n_files: int) -> None:
    """集計結果を .npz にキャッシュする（図の再描画を集計なしで行うため）。"""
    np.savez(path, a_n=c.a_n, a_s=c.a_s, m_c=c.m_c, m_s=c.m_s,
             meta=np.array([c.events, c.bad_handsize, n_files], dtype=np.int64))


def load_cache(path: Path) -> tuple[Counters, int]:
    """save_cache で保存した集計結果を読み戻す。"""
    z = np.load(path)
    c = Counters()
    c.a_n, c.a_s, c.m_c, c.m_s = z["a_n"], z["a_s"], z["m_c"], z["m_s"]
    c.events, c.bad_handsize, n_files = (int(x) for x in z["meta"])
    return c, n_files


def aggregate_game(events: list[dict[str, object]], c: Counters, no_reach: bool) -> None:
    """1 半荘を再構成し、序盤の切りの同半分クラス所持と位置別マージナルを積算する。"""
    hands: list[Counter[str]] = []
    melds: list[int] = []
    dcount: list[int] = []
    reach_active = False

    def start(ev: dict[str, object]) -> None:
        nonlocal hands, melds, dcount, reach_active
        tehais = cast("list[list[str]]", ev["tehais"])
        hands = [Counter(canon(t) for t in th) for th in tehais]
        melds = [0, 0, 0, 0]
        dcount = [0, 0, 0, 0]
        reach_active = False

    def remove(actor: int, tiles: list[str]) -> None:
        for t in tiles:
            cc = canon(t)
            hands[actor][cc] -= 1
            if hands[actor][cc] <= 0:
                del hands[actor][cc]

    for ev in events:
        t = cast(str, ev["type"])
        if t == "start_kyoku":
            start(ev)
        elif not hands:
            continue
        elif t == "tsumo":
            hands[cast(int, ev["actor"])][canon(cast(str, ev["pai"]))] += 1
        elif t == "dahai":
            a = cast(int, ev["actor"])
            p = canon(cast(str, ev["pai"]))
            remove(a, [p])
            dcount[a] += 1
            j = dcount[a]
            if sum(hands[a].values()) != 13 - 3 * melds[a]:
                c.bad_handsize += 1
            nm = parse_num(p)
            if nm is None or (no_reach and reach_active) or j > N_J:
                continue
            sx, raw = nm
            hd = hands[a]
            ji = j - 1
            # マージナル（全切りの手牌の位置別 P>=1, 全色合算 / 3色）
            c.m_c[ji] += 3
            for tile, _cn in hd.items():
                q = parse_num(tile)
                if q is not None:
                    c.m_s[ji, q[1]] += 1
            # signal: 切った牌と同じ半分の cy クラス牌の所持
            low = raw <= 5
            cx = raw if low else 10 - raw
            c.events += 1
            c.a_n[ji, cx - 1] += 1
            for cy in range(1, N_CLASS + 1):
                if cx == 5:  # 中心牌は左右対称なので両側平均
                    h = 0.5 * ((hd.get(f"{cy}{sx}", 0) >= 1)
                               + (hd.get(f"{10 - cy}{sx}", 0) >= 1))
                else:
                    pos = cy if low else 10 - cy
                    h = 1.0 if hd.get(f"{pos}{sx}", 0) >= 1 else 0.0
                c.a_s[ji, cx - 1, cy - 1] += h
        elif t == "reach_accepted":
            reach_active = True
        elif t in ("pon", "chi", "daiminkan"):
            remove(cast(int, ev["actor"]), cast("list[str]", ev["consumed"]))
            melds[cast(int, ev["actor"])] += 1
        elif t == "ankan":
            remove(cast(int, ev["actor"]), cast("list[str]", ev["consumed"]))
            melds[cast(int, ev["actor"])] += 1
        elif t == "kakan":
            remove(cast(int, ev["actor"]), cast("list[str]", ev["consumed"]))
        elif t in ("end_kyoku", "end_game"):
            hands = []


def block(c: Counters, cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """j<=cutoff をプールした (切りクラス×見るクラス) の 単純所持率・構造baseline・件数。"""
    sl = slice(0, cutoff)
    an = c.a_n[sl].astype(np.float64)       # (j, cx)
    as_ = c.a_s[sl].astype(np.float64)      # (j, cx, cy)
    mc = c.m_c[sl].astype(np.float64)       # (j,)
    ms = c.m_s[sl].astype(np.float64)       # (j, 10)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(mc[:, None] > 0, ms / np.maximum(mc[:, None], 1), np.nan)  # (j,10)
    nsum = an.sum(axis=0)                    # (cx,)
    sig = np.where(nsum[:, None] > 0, as_.sum(axis=0) / np.maximum(nsum[:, None], 1), np.nan)
    base = np.full((N_CLASS, N_CLASS), np.nan)
    for cx in range(1, N_CLASS + 1):
        if nsum[cx - 1] == 0:
            continue
        w = an[:, cx - 1]
        for cy in range(1, N_CLASS + 1):
            bj = 0.5 * (mp[:, cy] + mp[:, 10 - cy])   # 位置 cy の鏡平均マージナル
            valid = ~np.isnan(bj) & (w > 0)
            if valid.any():
                base[cx - 1, cy - 1] = (w[valid] * bj[valid]).sum() / w[valid].sum()
    return sig, base, nsum


def _heatmap(mat: np.ndarray, title: str, fname: str, outdir: Path, *,
             cmap: str, center: float | None) -> None:
    """5×5 行列を seaborn で1枚のヒートマップとして保存する（左下起点 1/9〜5）。"""
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    mat_flipped = mat[::-1, :]
    yticklabels_flipped = CLASS_LABEL[::-1]
    sns.heatmap(
        mat_flipped, ax=ax, cmap=cmap, center=center, annot=True, fmt=".3f",
        xticklabels=CLASS_LABEL, yticklabels=yticklabels_flipped,
        linewidths=0.5, linecolor="white", square=True,
        annot_kws={"fontsize": 11}, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("所持を確認する牌（対象牌）")
    ax.set_ylabel("切った牌の種類")
    ax.set_title(title)
    ax.tick_params(rotation=0)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render(c: Counters, tag: str, outdir: Path) -> None:
    """j<=CUTOFF の 単純所持率 と baseline との差分 を 2 枚のヒートマップで描く。"""
    sig, base, _ = block(c, CUTOFF)
    _heatmap(sig, f"条件付き所持率 (j≤{CUTOFF})", f"heatmap_rate_{tag}.png",
             outdir, cmap="Reds", center=None)
    _heatmap(sig - base, f"ベースラインとの差 (j≤{CUTOFF})", f"heatmap_diff_{tag}.png",
             outdir, cmap="RdBu_r", center=0.0)

def _line_plot(values: np.ndarray, title: str, ylabel: str, fname: str, outdir: Path, *,
               ylim: tuple[float, float | None], hline: float | None = None,
               max_j: int = 12) -> None:
    """(巡目 × 切りクラス) の折れ線を 1 枚保存する（列 cv-1 が切りクラス cv）。"""
    js = np.arange(1, N_J + 1)
    palette = sns.color_palette("tab10", N_CLASS)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for cv in range(1, N_CLASS):        # cv=1..4（内側に1つ隣 cv+1 が存在）
        ax.plot(js, values[:, cv - 1], "o-", color=palette[cv - 1], lw=1.8, ms=4,
                label=f"{CLASS_LABEL[cv - 1]}（1つ内側＝{CLASS_LABEL[cv]}）")
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8)
    if ylim[1] is None:
        ax.set_ylim(bottom=ylim[0])
    else:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(0.5, max_j + 0.5)
    ax.set_xticks(range(1, max_j + 1))
    ax.set_xlabel("切られた巡目 n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="切った牌の種類")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_line(c: Counters, tag: str, outdir: Path, max_j: int = 12) -> None:
    """1つ内側（中心方向）の所持を巡目ごとに折れ線で描く（条件付き所持率とリフトの 2 枚）。

    条件付き所持率 = P(1つ内側を所持 | n 巡目ちょうどにその牌を切った)。
    リフト（持っていない度）= baseline − 条件付き所持率（正。高いほど読みが強い）。
    巡目は非累積。早い巡目ほど所持率が低く、遅くなると baseline に近づく。
    """
    mc = c.m_c.astype(np.float64)
    ms = c.m_s.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(mc[:, None] > 0, ms / np.maximum(mc[:, None], 1), np.nan)  # (N_J,10)
    cond = np.full((N_J, N_CLASS - 1), np.nan)
    lift = np.full((N_J, N_CLASS - 1), np.nan)
    for cv in range(1, N_CLASS):
        cy = cv + 1
        for j in range(N_J):
            n = c.a_n[j, cv - 1]
            if n > 0 and mc[j] > 0:
                sig = c.a_s[j, cv - 1, cy - 1] / n
                base = 0.5 * (mp[j, cy] + mp[j, 10 - cy])
                cond[j, cv - 1] = sig
                lift[j, cv - 1] = base - sig
    _line_plot(cond, "巡目ごとの条件付き所持率",
               "1つ内側の牌の所持率", f"line_inner_cond_{tag}.png", outdir,
               ylim=(0.0, 0.35), max_j=max_j)
    _line_plot(lift, f"1つ内側を持っていない度合い：切られた巡目が早いほど高い  ({tag})",
               "1つ内側を持っていない度合い（基準からの低下幅）",
               f"line_inner_by_turn_{tag}.png", outdir,
               ylim=(-0.02, None), hline=0.0, max_j=max_j)


def _fmt(mat: np.ndarray, signed: bool) -> list[str]:
    """5×5 行列を等幅テキスト行にする。"""
    head = f"{'切\\見':>6} " + " ".join(f"{c:>7}" for c in CLASS_LABEL)
    lines = [head]
    for i in range(N_CLASS):
        cells = " ".join(
            (f"{mat[i, k]:>+7.3f}" if signed else f"{mat[i, k]:>7.3f}")
            if not np.isnan(mat[i, k]) else f"{'-':>7}"
            for k in range(N_CLASS))
        lines.append(f"{CLASS_LABEL[i]:>6} " + cells)
    return lines


def print_report(c: Counters, tag: str, n_files: int) -> None:
    """cutoff ごとに 単純所持率・構造baseline・リフトの 5×5 を出力する。"""
    print(f"# 序盤の切りのそばの所持（切りクラス×見るクラス 5×5）  {tag}  files={n_files}")
    print(f"# 対象切り={c.events}  handsize不整合={c.bad_handsize}")
    for cutoff in CUTOFFS_TABLE:
        sig, base, nsum = block(c, cutoff)
        print(f"\n===== 序盤 j<={cutoff}  （切りクラス別 n: "
              + " ".join(f"{CLASS_LABEL[i]}={int(nsum[i])}" for i in range(N_CLASS)) + "） =====")
        print("\n## 指標1 単純所持率")
        print("\n".join(_fmt(sig, signed=False)))
        print("\n## 構造baseline")
        print("\n".join(_fmt(base, signed=False)))
        print("\n## 指標2 リフト(=単純−baseline)")
        print("\n".join(_fmt(sig - base, signed=True)))


def run(data_dir: Path, outdir: Path, n_games: int, no_reach: bool,
        from_cache: bool) -> None:
    """データを走査し（または集計キャッシュから）、序盤の切りのそばの所持を出力する。"""
    setup_japanese_font()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "report").mkdir(parents=True, exist_ok=True)
    tag = f"{data_dir.name}_{'noreach' if no_reach else 'allreach'}"
    cache = outdir / f"counters_{tag}.npz"

    if from_cache and cache.exists():
        c, n_files = load_cache(cache)
        print(f"[cache] {cache} から読み込み（集計スキップ）")
    else:
        files = sorted(data_dir.glob("*.mjson"))
        if n_games:
            files = files[:n_games]
        c = Counters()
        for i, path in enumerate(files):
            try:
                aggregate_game(load_events(path), c, no_reach)
            except (OSError, json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
            if (i + 1) % 5000 == 0:
                print(f"... {i + 1}/{len(files)}", flush=True)
        n_files = len(files)
        save_cache(cache, c, n_files)

    render(c, tag, outdir / "figures")
    render_line(c, tag, outdir / "figures")
    print_report(c, tag, n_files)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="序盤の切りのそばの所持（切りクラス×見るクラス 5×5）")
    p.add_argument("--data-dir", type=Path, required=True, help="*.mjson を含むディレクトリ")
    p.add_argument("--outdir", type=Path, required=True, help="出力先")
    p.add_argument("--n-games", type=int, default=30000, help="対象局数（0 で全件）")
    p.add_argument("--no-reach", action="store_true", help="リーチ受理済みの観測を除外")
    p.add_argument("--from-cache", action="store_true",
                   help="集計をスキップし counters_*.npz から図・表を再生成する")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.outdir, args.n_games, args.no_reach, args.from_cache)
