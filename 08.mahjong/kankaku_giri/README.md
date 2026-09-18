# kankaku_giri — 間隔切り

セオリー「9 を切っている人が、間隔をあけて 8 を手出ししたら 6 を持っている」の定量化。
1 つ目を切ってから間隔をあけて 2 つ目を手出しする 4 ペア（12 / 24 / 35 / 46）を同じ枠組みで測る。
いずれも注目するのは **1 つ目に切った牌のスジ（3 つ内側）** の所持。
あわせて「間に字牌を挟むと確度が上がる」というセオリーを、間隔の分布を揃えたうえで検証する。

解析の意図・方法・結果・考察は [report/report.md](report/report.md) にある。

## 構成

```
scripts/kankaku_giri.py        集計本体（走査 → 表 → CSV → キャッシュ）
scripts/figs.py                キャッシュから図を作る
report/report.md               レポート（意図 / 方法 / 結果 / 考察）
report/run_noreach_full.txt    2025年全対局の実行ログ（表の出典）
figures/*.png                  図 5 枚
figures/*.csv                  ペア × 数ごとの所持率
counters_2025_noreach.npz      集計結果のキャッシュ（図の作り直し用）
```

## 実行

牌譜データは大容量（9.4GB）のため git 管理外。
天鳳・鳳凰卓 MJAI 2025年全対局 178,888半荘を `~/WorkSpace/blog/08.mahjong/data/2025` に展開してある。

```bash
cd ~/WorkSpace/blog/08.mahjong/kankaku_giri

# フル集計（全対局。16 プロセスで約 3 分）
python3 scripts/kankaku_giri.py \
  --data-dir ~/WorkSpace/blog/08.mahjong/data/2025 \
  --no-reach --workers 16 \
  --cache counters_2025_noreach.npz \
  --out-csv figures/kankaku_giri_2025_noreach.csv

# 表だけ作り直す（牌譜を読まない）
python3 scripts/kankaku_giri.py --from-cache --cache counters_2025_noreach.npz

# 図だけ作り直す（同梱の npz を使うのでデータ不要）
cd scripts && python3 figs.py --cache ../counters_2025_noreach.npz --outdir ../figures
```

- `--limit N` で先頭 N ファイルのみ（既定 0 = 全件）。
- `--no-reach` は「2 つ目の手出しの時点で誰か1人でもリーチ受理済みなら除外」。
  外すとリーチ後の強制ツモ切り・ベタオリが混ざる。
- 標準出力は門前のみ → 副露込みの順に 2 回出る。表の主指標は門前のみ。
  各回とも「y 別の所持率」「間に字牌を挟んだか」「間隔の長さ別」の 3 つの表が出る。
- 字牌は手出し / ツモ切りで所持率に差が出なかったため、表・図では有無だけで分けている。
  集計は内訳を保持しており、字牌別の表に参考列として出る。
- `figs.py` は `kankaku_giri.py` を import するので、同じディレクトリに置いたまま実行する。
- 日本語フォントはシステムの Noto Sans CJK JP
  （`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`）を直接登録している。
  japanize-matplotlib は使わない。
- 図の色は dataviz の参照パレット（検証済み）。4 系列の折れ線は直接ラベルを併用する。

## 実装メモ

- MJAI の `kakan` は `consumed` に**既にポン済みの 3 枚**が入り、手牌から出るのは `pai` の 1 枚だけ。
  DS 側の既存スクリプトは consumed 3 枚を手牌から引いており、`Counter` で 0 以下のキーを消す書き方の
  副作用で結果的に辻褄が合っていた（数値への影響は無く、既存実装と全量で一致することを確認済み）。
  ここでは `pai` を 1 枚引く実装にしてある。打牌ごとの `13 - 3 * melds` 検証が全件で 0 件。
- 集計は「同色 9 牌の所持を 9bit マスクにして数える」方式で、観測 1 件あたり 1 回の加算で済ませている。
  牌ごとの所持率は最後にビット表との行列積でほどく。
