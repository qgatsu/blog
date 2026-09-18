"""需給調整市場 (一次調整力) の約定結果を東京エリアの時系列に整形する。

データ取得: EPRX の取引実績ページは免責事項への同意 (POST) が必要なため、
手動または以下の手順で data/raw_eprx/ に zip を置く。

    curl -c cj.txt "https://www.eprx.or.jp/information/agree_results.php" -o /dev/null
    curl -L -b cj.txt -c cj.txt --data-urlencode "check=yes" \
         --data-urlencode "submit=取引実績へ" \
         "https://www.eprx.or.jp/information/results.php" -o /dev/null
    curl -b cj.txt -e "https://www.eprx.or.jp/information/results.php" \
         "https://www.eprx.or.jp/information/2026_1-0_result.zip" -o data/raw_eprx/2026_1-0_result.zip

制度の注意:
  - 2026年4月に全商品が前日取引化され、同時に 48 ブロック (30分) 単位になった。
    それ以前は週間商品・8 ブロック (3時間) 単位で、制度が異なる。
  - 価格の単位は 円/kW・30分。現行は上げ区分のみ調達。
  - 入札は実需給前日 11:30-14:00 締切、15時までに約定。
    JEPX スポット (前日 10:00 締切 / 10:30 約定) より後ろにある。
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_EPRX = DATA_DIR / "raw_eprx"

AREA_COL = 5          # 北海道,東北,東京,... の並びで東京は 5 列目 (0-origin)
WANTED = {
    "平均落札価格（TSO別）": "reserve_price",
    "落札量合計（TSO別）": "cleared_mw",
    "募集量（TSO別）": "required_mw",
    "応札量合計（電源属地別）": "offered_mw",
}


def parse_csv(text: str) -> list[tuple]:
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    nblock = int(lines[1].split(",")[4])
    out = []
    for ln in lines:
        p = ln.split(",")
        if len(p) <= AREA_COL or not re.match(r"^\d{8}B\d{2}$", p[0]):
            continue
        if p[1] != "システム約定結果":
            continue
        item = p[2].split("[")[0]
        if item not in WANTED:
            continue
        try:
            val = float(p[AREA_COL])
        except ValueError:
            val = np.nan
        out.append((p[0][:8], int(p[0][9:]), nblock, WANTED[item], val))
    return out


def main() -> None:
    rows = []
    for zp in sorted(RAW_EPRX.glob("*_1-0_result.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith(".csv"):
                    continue
                raw = z.read(name)
                for enc in ("cp932", "utf-8-sig"):
                    try:
                        text = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                rows += parse_csv(text)
    if not rows:
        raise SystemExit("データがありません。data/raw_eprx/ に zip を置いてください。")

    df = pd.DataFrame(rows, columns=["date", "block", "nblock", "item", "v"])
    wide = df.pivot_table(index=["date", "block", "nblock"], columns="item", values="v").reset_index()
    wide["date"] = pd.to_datetime(wide["date"])

    # 30分グリッドへ展開 (3時間ブロックの期間は同じ値を 6 コマに複製)
    expanded = []
    for _, r in wide.iterrows():
        slots = int(48 / r.nblock)
        start = int((r.block - 1) * slots)
        for k in range(slots):
            expanded.append({
                "datetime": r.date + pd.Timedelta(minutes=30 * (start + k)),
                "nblock": r.nblock,
                **{c: r[c] for c in WANTED.values() if c in wide.columns},
            })
    out = pd.DataFrame(expanded).set_index("datetime").sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # 現行制度 (前日取引・30分単位) のみを別途保存
    current = out[out.nblock == 48].drop(columns=["nblock"])

    out.to_parquet(DATA_DIR / "reserve_primary_tokyo_all.parquet")
    current.to_parquet(DATA_DIR / "reserve_primary_tokyo.parquet")

    meta = {
        "unit": "円/kW・30分",
        "area": "東京",
        "product": "一次調整力 (上げのみ)",
        "gate_closure": "実需給前日 11:30-14:00 入札 / 15時までに約定",
        "note": "2026年4月に全商品前日化・48ブロック化。それ以前は週間商品・8ブロック",
        "all_period": [str(out.index.min()), str(out.index.max())],
        "current_period": [str(current.index.min()), str(current.index.max())],
        "current_rows": len(current),
    }
    (DATA_DIR / "reserve_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"全期間     : {out.index.min().date()} ~ {out.index.max().date()}  {len(out):,} コマ")
    print(f"現行制度分 : {current.index.min().date()} ~ {current.index.max().date()}  {len(current):,} コマ")
    print(f"\n現行制度分の統計:\n{current.describe().round(3).to_string()}")


if __name__ == "__main__":
    main()
