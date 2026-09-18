"""捨て牌（河）の図を PNG で組む。

記法
    1 牌 = 1 トークン。空白区切りで並べる。
        牌     MJAI 表記（`1m`〜`9m` / `1p`〜`9p` / `1s`〜`9s` / `E S W N P F C`）。
               赤5は `5mr` `5pr` `5sr`。
        `*`    手出し。無印はツモ切りで、既定ではグレーアウトする。
        `<`    横向き（リーチ宣言牌）。
    例: `9p 1s E* 5mr 2m*< 7s`

段組
    既定で 6 枚ごとに改行する。横向きの牌は幅を余分に取るため、その段だけ右に伸びる。

使い方
    python3 sutehai.py "9p 1s E* 5mr 2m*< 7s 3p" -o out.png
    python3 sutehai.py "9p 1s E* 5mr" --highlight 2 --bg white -o out.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

TILE_DIR: Path = Path(__file__).resolve().parent.parent / "assets" / "tiles"

HONOR_FILES: dict[str, str] = {
    "E": "Ton", "S": "Nan", "W": "Shaa", "N": "Pei",
    "P": "Haku", "F": "Hatsu", "C": "Chun",
}
SUIT_FILES: dict[str, str] = {"m": "Man", "p": "Pin", "s": "Sou"}
FRONT_NAME: str = "Front"

TOKEN_RE: re.Pattern[str] = re.compile(r"^([1-9][mps]r?|[ESWNPFC])([*<]*)$")


class SutehaiError(Exception):
    """この module が投げる例外の基底。"""


@dataclass(frozen=True)
class Tile:
    """河に置かれた 1 牌。

    Attributes
    ----------
    name:
        牌画像のファイル名（拡張子なし）。
    tedashi:
        手出しなら True。ツモ切り（False）はグレーアウトして描く。
    sideways:
        リーチ宣言牌として横向きに置くなら True。
    """

    name: str
    tedashi: bool
    sideways: bool


@dataclass(frozen=True)
class Style:
    """描画の見た目。

    Attributes
    ----------
    tile_height:
        縦向きの牌の高さ（px）。幅は元画像の縦横比から決まる。
    gap:
        牌どうしの隙間（px）。
    margin:
        図の外周の余白（px）。
    per_row:
        1 段あたりの枚数。
    gray_target:
        グレーアウトの混ぜ先の色。
    gray_amount:
        ツモ切りのグレーアウトの強さ。0 で無効、1 で完全に gray_target。
    background:
        背景色。
    highlight:
        強調枠の色。
    """

    tile_height: int = 96
    gap: int = 7
    margin: int = 12
    per_row: int = 6
    gray_target: tuple[int, int, int] = (150, 150, 150)
    gray_amount: float = 0.55
    background: tuple[int, int, int, int] = (255, 255, 255, 0)
    highlight: tuple[int, int, int, int] = (222, 60, 60, 255)


def tile_filename(pai: str) -> str:
    """MJAI の牌表記を画像ファイル名（拡張子なし）に変換する。

    Parameters
    ----------
    pai:
        `1m` `5mr` `E` のような MJAI 表記。

    Raises
    ------
    SutehaiError
        未知の牌表記のとき。
    """
    if pai in HONOR_FILES:
        return HONOR_FILES[pai]
    if len(pai) >= 2 and pai[1] in SUIT_FILES and pai[0].isdigit():
        suit = SUIT_FILES[pai[1]]
        red = len(pai) == 3 and pai[2] == "r"
        return f"{suit}{pai[0]}-Dora" if red else f"{suit}{pai[0]}"
    raise SutehaiError(f"未知の牌: {pai!r}")


def parse(spec: str) -> list[Tile]:
    """捨て牌の記法を Tile の列に変換する。

    Parameters
    ----------
    spec:
        空白区切りのトークン列。

    Raises
    ------
    SutehaiError
        トークンが記法に合わないとき。
    """
    tiles: list[Tile] = []
    for token in spec.split():
        m = TOKEN_RE.match(token)
        if m is None:
            raise SutehaiError(f"読めないトークン: {token!r}")
        pai, flags = m.group(1), m.group(2)
        tiles.append(Tile(tile_filename(pai), "*" in flags, "<" in flags))
    return tiles


def _grayed(img: Image.Image, style: Style) -> Image.Image:
    """牌画像を灰色寄りに寄せる（不透明部分だけ）。"""
    flat = Image.new("RGBA", img.size, (*style.gray_target, 255))
    mixed = Image.blend(img.convert("RGBA"), flat, style.gray_amount)
    mixed.putalpha(img.getchannel("A"))
    return mixed


def _read(name: str) -> Image.Image:
    """牌の素材を 1 枚読む。"""
    path = TILE_DIR / f"{name}.png"
    if not path.exists():
        raise SutehaiError(f"牌画像が無い: {path}")
    with Image.open(path) as raw:
        return raw.convert("RGBA")


def _load(name: str, style: Style, tedashi: bool, sideways: bool) -> Image.Image:
    """1 牌を組み立てる（牌面に絵柄を重ね、グレーアウト・回転まで済ませる）。

    素材は絵柄だけが透過 PNG で入っているため、`Front`（牌の面）を下に敷く。
    """
    front = _read(FRONT_NAME)
    face = _read(name)
    img = Image.new("RGBA", front.size, (255, 255, 255, 0))
    img.alpha_composite(front)
    img.alpha_composite(face)
    w = max(1, round(img.width * style.tile_height / img.height))
    img = img.resize((w, style.tile_height), Image.LANCZOS)
    if not tedashi and style.gray_amount > 0:
        img = _grayed(img, style)
    if sideways:
        img = img.rotate(90, expand=True)
    return img


def _layout(tiles: list[Tile], style: Style) -> list[list[Tile]]:
    """per_row 枚ごとに段へ割る。"""
    return [tiles[i:i + style.per_row] for i in range(0, len(tiles), style.per_row)]


def render(tiles: list[Tile], style: Style, highlight: set[int] | None = None) -> Image.Image:
    """捨て牌の図を組んで 1 枚の画像にする。

    Parameters
    ----------
    tiles:
        並べる牌。捨てられた順。
    style:
        見た目の設定。
    highlight:
        枠で囲む牌の位置（0 始まり）。

    Returns
    -------
    Image.Image
        RGBA の画像。
    """
    marks = highlight or set()
    rows = _layout(tiles, style)
    images = [[_load(t.name, style, t.tedashi, t.sideways) for t in row] for row in rows]

    row_widths = [sum(im.width for im in row) + style.gap * max(0, len(row) - 1) for row in images]
    row_heights = [max((im.height for im in row), default=0) for row in images]
    width = max(row_widths, default=0) + style.margin * 2
    height = sum(row_heights) + style.gap * max(0, len(rows) - 1) + style.margin * 2

    canvas = Image.new("RGBA", (width, height), style.background)
    draw = ImageDraw.Draw(canvas)
    index = 0
    y = style.margin
    for row, row_h in zip(images, row_heights, strict=True):
        x = style.margin
        for im in row:
            canvas.alpha_composite(im, (x, y + row_h - im.height))
            if index in marks:
                draw.rectangle(
                    (x - 2, y + row_h - im.height - 2, x + im.width + 1, y + row_h + 1),
                    outline=style.highlight, width=3,
                )
            x += im.width + style.gap
            index += 1
        y += row_h + style.gap
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="捨て牌の図を PNG で組む")
    p.add_argument("spec", help="牌の並び。例: '9p 1s E* 5mr 2m*< 7s'")
    p.add_argument("-o", "--out", type=Path, default=Path("sutehai.png"), help="出力先")
    p.add_argument("--height", type=int, default=96, help="縦向きの牌の高さ(px)")
    p.add_argument("--per-row", type=int, default=6, help="1段あたりの枚数")
    p.add_argument("--gap", type=int, default=7, help="牌どうしの隙間(px)")
    p.add_argument("--gray", type=float, default=0.55,
                   help="ツモ切りのグレーアウトの強さ(0で無効)")
    p.add_argument("--highlight", type=int, nargs="*", default=[],
                   help="枠で囲む牌の位置(0始まり)")
    p.add_argument("--bg", default="none", help="背景色。'none' で透過")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bg = (255, 255, 255, 0) if args.bg == "none" else (*Image.new("RGB", (1, 1), args.bg).getpixel((0, 0)), 255)
    style = Style(tile_height=args.height, gap=args.gap, per_row=args.per_row,
                  gray_amount=args.gray, background=bg)
    img = render(parse(args.spec), style, set(args.highlight))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"{args.out}  {img.width}x{img.height}")


if __name__ == "__main__":
    main()
