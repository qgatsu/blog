import re

with open("/home/kohei/WorkSpace/blog/public/JEPX/01_jepx_spot_prediction.md", "r", encoding="utf-8") as f:
    text = f.read()

# Add space before ** if preceded by non-space/non-punctuation, and after ** if followed by non-space/non-punctuation
# It's safer to just do specific replacements
replacements = [
    ("取引する**「スポット市場（前日市場）」**です", "取引する **「スポット市場（前日市場）」** です"),
    ("区切った**計48コマ**の電力", "区切った **計48コマ** の電力"),
    ("制約として、**翌日", "制約として、 **翌日"),
    ("締め切られます**。", "締め切られます** 。"),
    ("市場において**「安い", "市場において **「安い"),
    ("売る」**ことで", "売る」** ことで"),
    ("ベースラインでは**LightGBM**、改善案では**MLP**を使用", "ベースラインでは **LightGBM** 、改善案では **MLP** を使用"),
    ("を用いた**線形計画法(LP)**を使用", "を用いた **線形計画法(LP)** を使用"),
    ("本記事は、**「1. 予測モデリング」**のアプローチ", "本記事は、 **「1. 予測モデリング」** のアプローチ"),
    ("にとどまらず、**「予測結果", "にとどまらず、 **「予測結果"),
    ("最終的な収益」**との関係性", "最終的な収益」** との関係性"),
    ("中での**「どの時間が", "中での **「どの時間が"),
    ("（ランキング）」**が決定的に", "（ランキング）」** が決定的に"),
    ("比較して**MAE（絶対値", "比較して **MAE（絶対値"),
    ("有意に改善**しました", "有意に改善** しました")
]

for old, new in replacements:
    text = text.replace(old, new)

with open("/home/kohei/WorkSpace/blog/public/JEPX/01_jepx_spot_prediction.md", "w", encoding="utf-8") as f:
    f.write(text)
