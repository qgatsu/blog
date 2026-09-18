"""MJAI 牌譜から、ある局・あるプレイヤーの捨て牌を取り出して sutehai.py の記法にする。

鳴かれた牌は河から取り除く（実際の卓上と同じ）。リーチ宣言牌は横向き `<`、手出しは `*` を付ける。

使い方
    python3 kawa_from_mjai.py <牌譜.mjson> --list
    python3 kawa_from_mjai.py <牌譜.mjson> --kyoku 1 --honba 0 --actor 2 -o out.png
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import sutehai

BAKAZE_JP: dict[str, str] = {"E": "東", "S": "南", "W": "西", "N": "北"}


@dataclass(frozen=True)
class KyokuInfo:
    """局の見出し。

    Attributes
    ----------
    index:
        牌譜内での通し番号（0 始まり）。
    bakaze:
        場風（`E` など）。
    kyoku:
        局数（1〜4）。
    honba:
        本場。
    reach_actors:
        その局でリーチしたプレイヤーの席番号。
    """

    index: int
    bakaze: str
    kyoku: int
    honba: int
    reach_actors: tuple[int, ...]

    def label(self) -> str:
        """`東1局0本場` のような表示名を返す。"""
        return f"{BAKAZE_JP.get(self.bakaze, self.bakaze)}{self.kyoku}局{self.honba}本場"


def load_events(path: Path) -> list[dict[str, object]]:
    """MJAI（NDJSON）を読む。gzip 圧縮されていても拡張子が .mjson のことがある。"""
    try:
        with gzip.open(path, "rt") as f:
            text = f.read()
    except OSError:
        text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def split_kyoku(events: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """イベント列を局ごとに切る。"""
    out: list[list[dict[str, object]]] = []
    cur: list[dict[str, object]] = []
    for ev in events:
        if ev["type"] == "start_kyoku":
            if cur:
                out.append(cur)
            cur = [ev]
        elif cur:
            cur.append(ev)
    if cur:
        out.append(cur)
    return out


def describe(kyokus: list[list[dict[str, object]]]) -> list[KyokuInfo]:
    """各局の見出しを作る。"""
    infos: list[KyokuInfo] = []
    for i, evs in enumerate(kyokus):
        head = evs[0]
        reach = tuple(sorted({cast(int, e["actor"]) for e in evs if e["type"] == "reach"}))
        infos.append(KyokuInfo(i, cast(str, head["bakaze"]), cast(int, head["kyoku"]),
                               cast(int, head["honba"]), reach))
    return infos


def kawa(events: list[dict[str, object]], actor: int) -> list[str]:
    """1 局ぶんのイベントから、そのプレイヤーの河を sutehai 記法のトークン列にする。

    Parameters
    ----------
    events:
        1 局ぶんのイベント（先頭が start_kyoku）。
    actor:
        席番号（0〜3）。

    Returns
    -------
    list[str]
        `2m*` `E<` のようなトークン。鳴かれた牌は含まない。
    """
    tokens: list[str] = []
    reach_pending = False
    for ev in events:
        t = cast(str, ev["type"])
        if t == "reach" and cast(int, ev["actor"]) == actor:
            reach_pending = True
        elif t == "dahai" and cast(int, ev["actor"]) == actor:
            flags = "" if ev.get("tsumogiri") else "*"
            if reach_pending:
                flags += "<"
                reach_pending = False
            tokens.append(f"{ev['pai']}{flags}")
        elif t in ("pon", "chi", "daiminkan") and cast(int, ev["target"]) == actor:
            # 直前に自分が捨てた牌が鳴かれた。河から取り除く。
            if tokens:
                tokens.pop()
    return tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MJAI 牌譜から捨て牌図を作る")
    p.add_argument("path", type=Path, help="牌譜ファイル(.mjson)")
    p.add_argument("--list", action="store_true", help="局の一覧を表示して終わる")
    p.add_argument("--index", type=int, help="局の通し番号(0始まり)")
    p.add_argument("--kyoku", type=int, help="局数で指定")
    p.add_argument("--honba", type=int, default=0, help="本場（--kyoku と併用）")
    p.add_argument("--actor", type=int, default=0, help="席番号(0〜3)")
    p.add_argument("-o", "--out", type=Path, help="PNG の出力先。省略時はトークンのみ表示")
    p.add_argument("--limit", type=int, help="河の先頭から何枚までを描くか")
    p.add_argument("--highlight", type=int, nargs="*", default=[],
                   help="枠で囲む牌の位置(0始まり)")
    p.add_argument("--height", type=int, default=96, help="牌の高さ(px)")
    p.add_argument("--gap", type=int, default=7, help="牌どうしの隙間(px)")
    p.add_argument("--bg", default="none", help="背景色。'none' で透過")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kyokus = split_kyoku(load_events(args.path))
    infos = describe(kyokus)

    if args.list:
        for info in infos:
            reach = "、".join(f"席{a}" for a in info.reach_actors) or "なし"
            print(f"[{info.index:2}] {info.label():<12} リーチ: {reach}")
        return

    if args.index is not None:
        picked = args.index
    elif args.kyoku is not None:
        match = [i.index for i in infos if i.kyoku == args.kyoku and i.honba == args.honba]
        if not match:
            raise SystemExit(f"該当する局が無い: {args.kyoku}局{args.honba}本場")
        picked = match[0]
    else:
        picked = 0

    tokens = kawa(kyokus[picked], args.actor)
    if args.limit is not None:
        tokens = tokens[:args.limit]
    print(f"# {infos[picked].label()}  席{args.actor}  {len(tokens)}枚")
    print(" ".join(tokens))

    if args.out is not None:
        bg = ((255, 255, 255, 0) if args.bg == "none"
              else (*sutehai.Image.new("RGB", (1, 1), args.bg).getpixel((0, 0)), 255))
        style = sutehai.Style(tile_height=args.height, gap=args.gap, background=bg)
        img = sutehai.render(sutehai.parse(" ".join(tokens)), style, set(args.highlight))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.out)
        print(f"{args.out}  {img.width}x{img.height}")


if __name__ == "__main__":
    main()
