"""間隔切りの集計結果から図を作る（kankaku_giri.py の集計を読む）."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import kankaku_giri as kg

plt.switch_backend("Agg")

_JP_FONT: Path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# dataviz の参照パレット（検証済み）
BLUE: str = "#2a78d6"  # categorical slot 1 / sequential 450
ORANGE: str = "#eb6834"  # slot 2
AQUA: str = "#1baf7a"  # slot 3
YELLOW: str = "#eda100"  # slot 4
RED: str = "#e34948"  # slot 8（diverging の暖色極）
SEQ: tuple[str, str, str] = ("#86b6ef", "#2a78d6", "#184f95")  # ordinal 3 段
INK: str = "#0b0b0b"
INK_SUB: str = "#52514e"
GRID: str = "#d8d7d2"
SERIES: tuple[str, str, str, str] = (BLUE, ORANGE, AQUA, YELLOW)
# 間隔を横軸に取る図の共通の上限。9 以降は「間に字牌なし」の件数が 500 を割り
# （間隔を大きくあけて字牌を1枚も挟まない打ち方が珍しいため）、所持率の信頼区間が
# ±0.06 まで広がる。2 枚の図で横軸を揃える意味も兼ねてここで打ち切る。
GAP_MAX_SHOWN: int = 8


def class_label(v: int) -> str:
    """canonical 数を端距離クラス表記にする（5 は鏡像が自分自身なので "5"）."""
    return "5" if v == 5 else f"{min(v, 10 - v)}/{max(v, 10 - v)}"


def pair_short(a: int, b: int) -> str:
    """線の端に添える短い表記."""
    return f"{a}→{b}"


def pair_full(a: int, b: int, target: int) -> str:
    """凡例用の表記。統合している鏡像を併記し、注目牌はクラスで書く."""
    return f"{a}→{b} / {10 - a}→{10 - b}（{class_label(target)} の所持）"


def pair_full_2line(a: int, b: int, target: int) -> str:
    """軸ラベル用。pair_full を 2 行に折る."""
    return f"{a}→{b} / {10 - a}→{10 - b}\n（{class_label(target)} の所持）"


def lighten(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """色を白に寄せて薄くする（同じ系列の中で 2 群を分けるため）."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def setup_font() -> None:
    """日本語フォントを matplotlib に登録する."""
    if _JP_FONT.exists():
        fm.fontManager.addfont(str(_JP_FONT))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = GRID
    plt.rcParams["text.color"] = INK
    plt.rcParams["axes.labelcolor"] = INK_SUB
    plt.rcParams["xtick.color"] = INK_SUB
    plt.rcParams["ytick.color"] = INK_SUB
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


@dataclass(frozen=True)
class PairResult:
    """1 ペア分の所持率（巡目標準化済み）."""

    a: int
    b: int
    cond: npt.NDArray[np.float64]
    control: npt.NDArray[np.float64]
    none_rate: npt.NDArray[np.float64]
    adj_rate: npt.NDArray[np.float64]
    honor: npt.NDArray[np.float64]  # (字牌なし / 字牌あり, y)
    honor_n: npt.NDArray[np.int64]
    n_cond: int
    n_ctrl: int
    n_adj: int

    @property
    def target(self) -> int:
        """注目する所持牌（1 つ目に切った牌のスジ）."""
        return self.a + 3

    @property
    def lift(self) -> npt.NDArray[np.float64]:
        """対照との差."""
        return self.cond - self.control


def compute_pairs(summary: kg.Summary) -> list[PairResult]:
    """集計から、ペアごとの所持率をまとめる."""
    out: list[PairResult] = []
    for pair_i, (a, b) in enumerate(kg.PAIRS):
        n_st = summary.n[pair_i]
        s_st = summary.sums[pair_i]
        totals = n_st.sum(axis=0)
        cond_n = n_st[:, kg.GAP_STATES].sum(axis=1)
        cond_s = s_st[:, kg.GAP_STATES, :].sum(axis=1)
        weight = cond_n.astype(np.float64)
        ctrl_n = n_st[:, kg.STATE_NONE] + n_st[:, kg.STATE_ADJACENT]
        ctrl_s = s_st[:, kg.STATE_NONE, :] + s_st[:, kg.STATE_ADJACENT, :]
        # 間に挟んだ字牌ごとの所持率は、間隔の分布を間隔あり全体に揃えて出す
        gn = summary.gap_n[pair_i]
        gs = summary.gap_sum[pair_i]
        gap_weight = gn.sum(axis=1).astype(np.float64)
        honor = np.stack(
            [
                kg.standardized_rate(gn[:, idx].sum(axis=1), gs[:, idx, :].sum(axis=1), gap_weight)
                for idx in (kg.GAP_HONOR_NONE, kg.GAP_HONOR_ANY)
            ]
        )
        out.append(
            PairResult(
                a=a,
                b=b,
                cond=cond_s.sum(axis=0) / max(cond_n.sum(), 1),
                control=kg.standardized_rate(ctrl_n, ctrl_s, weight),
                none_rate=kg.standardized_rate(n_st[:, kg.STATE_NONE], s_st[:, kg.STATE_NONE, :], weight),
                adj_rate=kg.standardized_rate(
                    n_st[:, kg.STATE_ADJACENT], s_st[:, kg.STATE_ADJACENT, :], weight
                ),
                honor=honor,
                honor_n=np.array(
                    [gn[:, kg.GAP_HONOR_NONE].sum(), gn[:, kg.GAP_HONOR_ANY].sum()],
                    dtype=np.int64,
                ),
                n_cond=int(cond_n.sum()),
                n_ctrl=int(ctrl_n.sum()),
                n_adj=int(totals[kg.STATE_ADJACENT]),
            )
        )
    return out


def render_lift_by_pair(pairs: list[PairResult], tag: str, path: Path) -> None:
    """ペアごとに、各数の所持率が対照からどれだけ動くかを描く."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharey=True)
    lim = max(float(np.nanmax(np.abs(p.lift))) for p in pairs) + 0.06
    for ax, p in zip(axes.ravel(), pairs):
        lift = p.lift
        colors = [RED if v >= 0 else BLUE for v in lift]
        ax.bar(range(1, 10), lift, color=colors, width=0.62)
        ax.axhline(0, color=INK_SUB, lw=0.9)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(1, 10))
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(
            f"{pair_full(p.a, p.b, p.target)}  n={p.n_cond:,}",
            fontsize=11,
            color=INK,
        )
        # 注目牌と、最も大きい負のセルだけ値を書く
        marked = {p.target, int(np.argmin(lift)) + 1}
        for y in marked:
            v = lift[y - 1]
            ax.text(
                y,
                v + (0.012 if v >= 0 else -0.012),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9.5,
                color=INK,
            )
        ax.annotate(
            "",
            xy=(p.target, 0),
            xytext=(p.target, lift[p.target - 1]),
            arrowprops={"arrowstyle": "-", "color": "none"},
        )
    for ax in axes[1]:
        ax.set_xlabel("手牌に持っているか見る牌")
    for ax in axes[:, 0]:
        ax.set_ylabel("対照との差（所持率）")
    fig.suptitle("間隔切りが起きたとき、どの牌の所持率が動くか", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_gap(summary: kg.Summary, pairs: list[PairResult], tag: str, path: Path) -> None:
    """間隔の長さごとに、注目牌の所持率がどう変わるかを描く."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    gaps = list(range(kg.MIN_GAP, GAP_MAX_SHOWN + 1))
    for pair_i, p in enumerate(pairs):
        xs: list[int] = []
        ys: list[float] = []
        for g in gaps:
            n = int(summary.gap_n[pair_i, g].sum())
            if n < 200:
                continue
            xs.append(g)
            ys.append(float(summary.gap_sum[pair_i, g, :, p.target - 1].sum() / n))
        ax.plot(
            xs,
            ys,
            "o-",
            color=SERIES[pair_i],
            lw=2.0,
            ms=6.5,
            label=pair_full(p.a, p.b, p.target),
        )
        ax.annotate(
            pair_short(p.a, p.b),
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=INK,
        )
        ax.axhline(p.control[p.target - 1], color=SERIES[pair_i], lw=1.8, ls=":", alpha=0.8)
    ax.set_xticks(gaps)
    ax.set_xlabel("間隔の大きさ")
    ax.set_ylabel("注目牌の所持率")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # 点線が何を指すかは凡例に出す（同色の水準線がペアごとに4本ある）
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK_SUB, lw=1.8, ls=":"))
    labels.append("間隔切りが無いときの水準")
    # 水準線と重なるので、凡例は半透明のボックスに載せる
    legend = ax.legend(
        handles,
        labels,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor=GRID,
        fontsize=9.5,
        loc="lower right",
        ncol=2,
    )
    legend.set_zorder(5)
    ax.set_title("間隔の大きさごとの所持率推移", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_states(pairs: list[PairResult], tag: str, path: Path) -> None:
    """先行なし / 間隔なし / 間隔あり の 3 段階で注目牌の所持率を並べる."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    labels = ("1 つ目を切っていない", "間隔なし（直後に 2 つ目）", "間隔あり")
    width = 0.26
    xs = np.arange(len(pairs), dtype=np.float64)
    series = [
        [p.none_rate[p.target - 1] for p in pairs],
        [p.adj_rate[p.target - 1] for p in pairs],
        [p.cond[p.target - 1] for p in pairs],
    ]
    for k, (vals, color, label) in enumerate(zip(series, SEQ, labels)):
        ax.bar(xs + (k - 1) * width, vals, width=width - 0.02, color=color, label=label)
    for i, p in enumerate(pairs):
        v = p.cond[p.target - 1]
        ax.text(
            xs[i] + width,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([pair_full_2line(p.a, p.b, p.target) for p in pairs])
    ax.set_ylabel("注目牌の所持率")
    ax.set_ylim(0, max(max(s) for s in series) + 0.09)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("間隔をあけたときだけ所持率が大きく上がる", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_honor(pairs: list[PairResult], tag: str, path: Path) -> None:
    """間に挟んだ字牌の種類ごとに、注目牌の所持率を並べる（間隔の分布は揃えてある）."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    labels = ("間に字牌なし", "間に字牌あり")
    width = 0.34
    xs = np.arange(len(pairs), dtype=np.float64)
    # 色は line_gap と同じペアごとの系列色。字牌の有無は濃淡で分ける
    for k, label in enumerate(labels):
        vals = [float(p.honor[k, p.target - 1]) for p in pairs]
        colors = [SERIES[i] if k == 1 else lighten(SERIES[i]) for i in range(len(pairs))]
        offset = (k - 0.5) * width
        ax.bar(xs + offset, vals, width=width - 0.03, color=colors, label=label)
        for i, v in enumerate(vals):
            ax.text(
                xs[i] + offset,
                v + 0.008,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_SUB,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([pair_full_2line(p.a, p.b, p.target) for p in pairs])
    ax.set_ylabel("注目牌の所持率")
    ax.set_ylim(0, max(float(p.honor[:, p.target - 1].max()) for p in pairs) + 0.10)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        Patch(facecolor=lighten(INK)),
        Patch(facecolor=INK),
    ]
    ax.legend(handles, list(labels), frameon=False, fontsize=9.5, loc="upper right", ncol=2)
    ax.set_title("間に字牌を挟んだかで分ける（間隔の分布は揃えてある）", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_gap_by_honor(
    summary: kg.Summary, pairs: list[PairResult], tag: str, path: Path
) -> None:
    """間隔の長さごとに、字牌を挟んだ場合と挟まない場合を並べる.

    両群とも十分な件数がある GAP_MAX_SHOWN までを描く（理由は定数のコメント）。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=True)
    gaps = list(range(kg.MIN_GAP, GAP_MAX_SHOWN + 1))
    for ax, (pair_i, p) in zip(axes.ravel(), enumerate(pairs)):
        # 系列の色は line_gap と同じ（ペアごと）。字牌の有無は濃淡で分ける
        base = SERIES[pair_i]
        for color, label, idx in (
            (lighten(base), "間に字牌なし", kg.GAP_HONOR_NONE),
            (base, "間に字牌あり", kg.GAP_HONOR_ANY),
        ):
            xs: list[int] = []
            ys: list[float] = []
            for g in gaps:
                n = int(summary.gap_n[pair_i, g, idx].sum())
                if n < 500:
                    continue
                xs.append(g)
                ys.append(float(summary.gap_sum[pair_i, g, idx, p.target - 1].sum() / n))
            ax.plot(xs, ys, "o-", color=color, lw=2.0, ms=6.0, label=label)
        ax.axhline(p.control[p.target - 1], color=INK_SUB, lw=1.8, ls=":", alpha=0.8)
        ax.set_title(pair_full(p.a, p.b, p.target), fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    # 濃淡の意味と点線の意味を、左上のパネルにまとめて出す。
    # 系列色はパネルごとに違うので、凡例は色を持たない黒と灰で濃淡の規則だけを示す
    handles = [
        Line2D([], [], color=lighten(INK), lw=2.0, marker="o", ms=6.0),
        Line2D([], [], color=INK, lw=2.0, marker="o", ms=6.0),
        Line2D([], [], color=INK_SUB, lw=1.8, ls=":"),
    ]
    labels = ["間に字牌なし", "間に字牌あり", "間隔切りが無いときの水準"]
    legend = axes[0, 0].legend(
        handles,
        labels,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor=GRID,
        fontsize=9.5,
        loc="lower right",
    )
    legend.set_zorder(5)
    for ax in axes[1]:
        ax.set_xlabel("間隔の大きさ")
    for ax in axes[:, 0]:
        ax.set_ylabel("注目牌の所持率")
    fig.suptitle("字牌が間に入っているかによる差分", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(cache: Path, outdir: Path, tag: str) -> None:
    """キャッシュから 3 枚の図を作る."""
    setup_font()
    part = kg.load_cache(cache)
    summary = kg.summarize(part, menzen_only=True)
    pairs = compute_pairs(summary)
    outdir.mkdir(parents=True, exist_ok=True)
    render_lift_by_pair(pairs, tag, outdir / f"bar_lift_by_pair_{tag}.png")
    render_gap(summary, pairs, tag, outdir / f"line_gap_{tag}.png")
    render_states(pairs, tag, outdir / f"bar_states_{tag}.png")
    render_honor(pairs, tag, outdir / f"bar_honor_{tag}.png")
    render_gap_by_honor(summary, pairs, tag, outdir / f"line_gap_by_honor_{tag}.png")
    print(f"[figures] {outdir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="間隔切りの図を作る")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="2025_noreach")
    a = ap.parse_args()
    main(a.cache, a.outdir, a.tag)
