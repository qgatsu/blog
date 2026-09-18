# joban_bakyou — 序盤の切りのそばは持たれていない

記事 `../../public/mahjong/02_joban_bakyou.md` の解析。
セオリー「序盤に切った牌のそばには周りの牌が無い」を、切った牌と見る牌を
端距離クラス（1/9, 2/8, 3/7, 4/6, 5）に畳んだ 5×5 で定量化する。

解析の意図・方法・結果・考察は [report/report.md](report/report.md) にある。

## 構成

```
scripts/joban_bakyou.py        解析本体（集計 → 図 → 表）
report/report.md               レポート（意図 / 方法 / 結果 / 考察）
report/run_noreach_full.txt    2025年全対局の実行ログ（表の出典）
figures/*.png                  記事で使う図 3 枚
counters_2025_noreach.npz      集計結果のキャッシュ（--from-cache 用）
```

## 実行

データは大容量（9.4GB）のためこのリポジトリには入っていない。
天鳳・鳳凰卓 MJAI 2025年全対局 178,888半荘を `~/WorkSpace/blog/08.mahjong/data/2025` に展開してある。

```bash
cd ~/WorkSpace/blog/08.mahjong/joban_bakyou

# フル集計（全対局。単一プロセスで約8分）
python3 scripts/joban_bakyou.py \
  --data-dir ~/WorkSpace/blog/08.mahjong/data/2025 \
  --outdir . --n-games 0 --no-reach

# 図・表だけ作り直す（同梱の npz を使うのでデータ不要）
python3 scripts/joban_bakyou.py \
  --data-dir ~/WorkSpace/blog/08.mahjong/data/2025 \
  --outdir . --n-games 0 --no-reach --from-cache
```

`--from-cache` でもデータ本体は読まないが `--data-dir` は必須で、
その **ディレクトリ名が出力ファイル名のタグ**（`2025_noreach`）になる。
名前が違うと同梱の npz を見つけられない。

- `--n-games 0` で全件。既定は 30000 局。
- `--no-reach` は「測定時点までに誰か1人でもリーチ受理済みなら除外」。
  外すと reach 後の強制ツモ切り・ベタオリが混ざる。
- 日本語フォントはシステムの Noto Sans CJK JP
  （`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`）を直接登録している。
  japanize-matplotlib は使わない。

## 出所

`~/WorkSpace/DS/mahjong/project/yomi-joban-bakyou/` からのコピー（2026-09-04 時点）。
DS 側にも同じものが残っている。同じ枠組みの派生解析（スジ・役牌・副露など）は
DS 側の `project/` にのみある。
