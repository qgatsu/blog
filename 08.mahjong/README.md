# 麻雀のセオリーを牌譜で検証する（シリーズ）

記事本体は `../public/mahjong/` にある. このディレクトリには解析プロジェクト・図・図の生成ツールを置く.

## 解析プロジェクトとデータ

解析は直下にプロジェクトフォルダとして置く（`<name>/{scripts,report,figures}`）.

- `joban_bakyou/` — 第2回の解析（`DS/mahjong/project/yomi-joban-bakyou/` からのコピー, DS 側も残っている）
- `kankaku_giri/` — 間隔切り（第3回「違和感のある手出し」の1つ目。記事化前）
- `gyakugiri/` — 中盤の 1/9・2/8 手出しと周りの牌（第3回の2つ目。記事化前）。
  逆切りを条件に測ろうとして成立しなかった経緯も同じレポートに残してある

第3回は「違和感のある手出し」を題材に, 間隔切り → 逆切り → 字牌/ドラの後の手出し を順に扱う.
DS 側の `project/` にも解析が残っているが, そちらは作り直す前提で参照しない.

牌譜データは `data/` に置く. 大容量かつ再配布しないので git 管理からは外してある（リポジトリルートの `.gitignore` で `08.mahjong/data/` と `*.mjson` を無視）.

- `~/WorkSpace/blog/08.mahjong/data/2025`（天鳳・鳳凰卓 MJAI, 178,888半荘, 9.4GB）

各プロジェクトの実行手順は `<name>/README.md` にある.

## 記事と図の対応

| 記事 | 図の置き場 | 使う図 |
|---|---|---|
| `01_intro.md` | — | なし |
| `02_joban_bakyou.md` | `joban_bakyou/figures/` | `heatmap_rate_2025_noreach.png`, `heatmap_diff_2025_noreach.png`, `line_inner_by_turn_2025_noreach.png` |
| `02_joban_bakyou.md`（セオリーの例） | `figures/`（`tools/` で生成） | `example_kawa.png`, `example_tehai.png` |

解析が生む図はプロジェクト配下の `figures/` に置く（解析をやり直せばその場で上書きされる）.
`tools/` で手作りする説明用の図だけ `figures/` 直下に置く.

## 捨て牌図の生成

`tools/sutehai.py`（記法から組む）と `tools/kawa_from_mjai.py`（牌譜から取り出す）。牌画像は `assets/tiles/`。

第2回の例は次で再現できる。

```bash
cd tools
rec=~/WorkSpace/blog/08.mahjong/data/2025/2025010118gm-00a9-0000-4314e7da.mjson
# 南3局0本場 席0 の河（6巡目まで）。第二打の8索が説明の対象
python3 kawa_from_mjai.py "$rec" --index 8 --actor 0 --limit 6 --bg white -o ../figures/example_kawa.png
# 8索を切った直後の手牌
python3 -c "import sutehai; \
  img=sutehai.render(sutehai.parse('4m* 5m* 8m* 9m* 2p* 2p* 3p* 4p* 7p* 8p* 9p* 2s* 2s*'), \
  sutehai.Style(per_row=13, background=(255,255,255,255))); img.save('../figures/example_tehai.png')"
```

第3回以降は解析をやり直してから決める.

## 公開手順

1. 記事を書く（frontmatter の `ignorePublish: true` のまま）
2. `npx qiita preview` で確認（http://127.0.0.1:8890 ）
3. 図を Qiita のエディタにドラッグしてアップロードし,得た S3 URL を記事の `画像URL` プレースホルダと差し替える
4. `ignorePublish` を消して main に push（GitHub Actions が publish する）

Qiita CLI は `public/` を再帰的に走査するので `public/mahjong/` 配下でも認識される（basename は `mahjong/01_intro`）. ただし `npx qiita new mahjong/xx` はディレクトリを作らないので,先に `mkdir` が要る.
