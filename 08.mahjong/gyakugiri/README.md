# 中盤の 1/9・2/8 手出しと、その周りの牌

セオリー「麻雀は外側から順に切るので、中盤以降に出てくる端牌はブロックに絡んでいる」の定量化。
打牌を **手出し / ツモ切り / 切っていない** の 3 状態に分け、切った牌から見た
**位置ごと**（1 つ外側 / 1 つ内側 / 2 つ内側）に、巡目別の所持率を測る。

解析の意図・方法・結果・考察は [report/report.md](report/report.md) にある。

もともと「逆切り」（内側を切った後の外側手出し）を条件に測ろうとしたが、
その条件が中盤以降ほぼ全員に当てはまり比較が成立しなかった。
経緯と数字はレポートの「逆切りという条件では測れなかった」に残してある。
ディレクトリ名はその名残。

## 構成

```
scripts/tedashi_window.py          集計本体（先行条件なし。現在の主解析）
scripts/figs_window.py             キャッシュから図を作る
scripts/gyakugiri.py               逆切りを条件にした集計（測れなかった版。経緯の記録）
report/report.md                   レポート（意図 / 方法 / 結果 / 考察）
report/run_window_full.txt         主解析の実行ログ（表の出典）
report/run_noreach_full.txt        逆切り版の実行ログ
figures/line_lift_by_pos_*.png     位置ごとのリフトの推移
figures/line_states_*.png          手出し / ツモ切り / 切っていない の推移
figures/bar_mid_*.png              中盤（巡目 7-12）の位置ごとのリフト
counters_window_2025_noreach.npz   主解析の集計キャッシュ
counters_2025_noreach.npz          逆切り版の集計キャッシュ
```

## 実行

牌譜データは大容量（9.4GB）のため git 管理外。
天鳳・鳳凰卓 MJAI 2025年全対局 178,888半荘を `~/WorkSpace/blog/08.mahjong/data/2025` に展開してある。

```bash
cd ~/WorkSpace/blog/08.mahjong/gyakugiri

# フル集計（全対局。16 プロセスで約 5 分）
python3 scripts/tedashi_window.py \
  --data-dir ~/WorkSpace/blog/08.mahjong/data/2025 \
  --no-reach --workers 16 \
  --cache counters_window_2025_noreach.npz

# 表だけ作り直す（牌譜を読まない）
python3 scripts/tedashi_window.py --from-cache --cache counters_window_2025_noreach.npz

# 図だけ作り直す
cd scripts && python3 figs_window.py \
  --cache ../counters_window_2025_noreach.npz --outdir ../figures
```

- `--no-reach` は「その打牌の時点で誰か1人でもリーチ受理済みなら除外」。
- `--min-n` はその件数未満の巡目を表から伏せる（既定 5000）。
- 出力は「位置ごと・巡目ごとの 3 状態」「中盤まとめ（2 指標）」「手出し / ツモ切りの平均リフト」。
- 日本語フォントはシステムの Noto Sans CJK JP を直接登録している。japanize-matplotlib は使わない。
- 図の色は dataviz の参照パレット（検証済み）。

## 実装メモ

- **base の作り方**: 打牌のたびに 3 色それぞれの「同色 9 牌の所持マスク」を数えておき、
  切られた位置の分を引いて「切っていない」を作る。位置ごとに全打牌を走査すると重すぎるため。
- **完成面子の判定**: `free_mask()` は「その牌を 1 枚抜いても最大面子数が変わらない」位置だけを立てる。
  567 の 7 は抜くと面子が崩れるので落ち、77 や 79 は残る。最大面子数は再帰で求めてメモ化している
  （メモ化が効いて全量でも数分）。この判定ではスライドが拾えない。
- **MJAI の `kakan`**: `consumed` に既にポン済みの 3 枚が入り、手牌から出るのは `pai` の 1 枚だけ。
  打牌ごとの `13 - 3 * melds` 検証が全件で 0 件になることを確認している。
