"""中盤の 1/9・2/8 手出しの図を作る（tedashi_window.py の集計を読む）."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

import tedashi_window as tw

plt.switch_backend("Agg")

_JP_FONT: Path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# dataviz の参照パレット（検証済み）
BLUE: str = "#2a78d6"
ORANGE: str = "#eb6834"
AQUA: str = "#1baf7a"
RED: str = "#e34948"
GRAY: str = "#8a8984"
INK: str = "#0b0b0b"
INK_SUB: str = "#52514e"
GRID: str = "#d8d7d2"

MID: range = range(6, 12)  # 中盤 = 巡目 7-12（打牌 index 6-11）


def setup_font() -> None:
    """日本語フォントと基本の見た目を整える."""
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


def series(rates: tw.Rates, cls: int, d: int, state: int, turns: range) -> list[float]:
    """クラス cls をオフセット d から見たときの、巡目ごとの所持率."""
    members = [cls] if cls == 5 else [cls, 10 - cls]
    out: list[float] = []
    for j in turns:
        hit = 0.0
        obs = 0.0
        for v in members:
            u = tw.offset_positions(v, d)
            if u is None:
                return []
            h, n = tw.rate_of(rates, v, j, state, u)
            hit += h
            obs += n
        out.append(hit / obs if obs else float("nan"))
    return out


def render_lift(rates: tw.Rates, tag: str, path: Path) -> None:
    """位置ごとのリフトが巡目でどう動くかを描く."""
    turns = range(0, 18)
    xs = [j + 1 for j in turns]
    specs = (
        (1, "1/9 を手出し", ((1, "1つ内側", BLUE), (2, "2つ内側", ORANGE))),
        (
            2,
            "2/8 を手出し",
            ((-1, "1つ外側", AQUA), (1, "1つ内側", BLUE), (2, "2つ内側", ORANGE)),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, (cls, title, items) in zip(axes, specs):
        for d, label, color in items:
            ted = series(rates, cls, d, 0, turns)
            base = series(rates, cls, d, 2, turns)
            lift = [t - b for t, b in zip(ted, base)]
            ax.plot(xs, lift, "o-", color=color, lw=2.0, ms=5.5, label=label)
            ax.annotate(
                label,
                xy=(xs[-1], lift[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=color,
            )
        ax.axhline(0, color=INK_SUB, lw=0.9)
        ax.axvspan(7, 12, color=GRID, alpha=0.35, lw=0)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("巡目")
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    axes[0].set_ylabel("手出しのリフト（切っていない との差）")
    fig.suptitle(
        f"手出しで所持率がどれだけ動くか（網掛け = 中盤 巡目7-12）  {tag}",
        fontsize=12.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_states(rates: tw.Rates, tag: str, path: Path) -> None:
    """手出し・ツモ切り・切っていない の 3 本を代表的な位置について描く."""
    turns = range(0, 18)
    xs = [j + 1 for j in turns]
    panels = (
        (1, 2, "1/9 を手出し → 2つ内側", "正に出る例"),
        (2, 1, "2/8 を手出し → 1つ内側", "負に出る例"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, (cls, d, title, note) in zip(axes, panels):
        for state, label, color in (
            (0, "手出し", BLUE),
            (1, "ツモ切り", ORANGE),
            (2, "切っていない（base）", GRAY),
        ):
            ys = series(rates, cls, d, state, turns)
            ax.plot(xs, ys, "o-", color=color, lw=2.0, ms=5.5, label=label)
        ax.axvspan(7, 12, color=GRID, alpha=0.35, lw=0)
        ax.set_title(f"{title}（{note}）", fontsize=11.5, color=INK)
        ax.set_xlabel("巡目")
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    axes[0].set_ylabel("その位置の牌を持っている割合")
    fig.suptitle(
        f"ツモ切りは一貫して base を下回る（網掛け = 中盤 巡目7-12）  {tag}",
        fontsize=12.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_mid(rates: tw.Rates, tag: str, path: Path) -> None:
    """中盤に絞って、位置ごとのリフトを並べる."""
    items = (
        (1, 1, "1/9 →\n1つ内側"),
        (1, 2, "1/9 →\n2つ内側"),
        (2, -1, "2/8 →\n1つ外側"),
        (2, 1, "2/8 →\n1つ内側"),
        (2, 2, "2/8 →\n2つ内側"),
    )
    labels: list[str] = []
    lifts: list[float] = []
    conds: list[float] = []
    bases: list[float] = []
    for cls, d, label in items:
        ted = series(rates, cls, d, 0, MID)
        base = series(rates, cls, d, 2, MID)
        n_ted = [tw.rate_of(rates, cls, j, 0, 1)[1] for j in MID]
        w = np.array(n_ted, dtype=np.float64)
        c = float(np.average(ted, weights=w))
        b = float(np.average(base, weights=w))
        labels.append(label)
        conds.append(c)
        bases.append(b)
        lifts.append(c - b)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [RED if v >= 0 else BLUE for v in lifts]
    ax.bar(range(len(labels)), lifts, color=colors, width=0.56)
    ax.axhline(0, color=INK_SUB, lw=0.9)
    for i, (v, c, b) in enumerate(zip(lifts, conds, bases)):
        ax.text(
            i,
            v + (0.006 if v >= 0 else -0.006),
            f"{v:+.3f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10,
            color=INK,
        )
        ax.text(
            i,
            0.004 if v < 0 else -0.004,
            f"{c:.3f} / {b:.3f}",
            ha="center",
            va="bottom" if v < 0 else "top",
            fontsize=8.5,
            color=INK_SUB,
        )
    span = max(lifts) - min(lifts)
    ax.set_ylim(min(lifts) - span * 0.18, max(lifts) + span * 0.15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("手出しのリフト（切っていない との差）")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        f"中盤（巡目7-12）の位置ごとのリフト  小さい数字 = 手出し / base  {tag}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(cache: Path, outdir: Path, tag: str) -> None:
    """キャッシュから図を作る."""
    setup_font()
    part = tw.load_cache(cache)
    rates = tw.compute(part)
    outdir.mkdir(parents=True, exist_ok=True)
    render_lift(rates, tag, outdir / f"line_lift_by_pos_{tag}.png")
    render_states(rates, tag, outdir / f"line_states_{tag}.png")
    render_mid(rates, tag, outdir / f"bar_mid_{tag}.png")
    print(f"[figures] {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="中盤の端牌手出しの図")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="2025_noreach")
    a = ap.parse_args()
    main(a.cache, a.outdir, a.tag)
