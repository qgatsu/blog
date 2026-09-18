# recommend_ml

MovieLens 1M を使った top-N 推薦の手法比較。基礎的な協調フィルタリングを
同一の評価プロトコルで実装し、どこまでの数値が出るのかを確認する。

## 何をやっているか

推薦の論文は前処理・分割・指標の定義が論文ごとに違い、数値をそのまま比較できない
ことが多い。また再現性研究 (Dacrema et al. 2019、Rendle et al. 2020 など) は
「ベースラインをきちんとチューニングすると新しい手法を上回る」例を繰り返し
報告している。そこでここでは、

- 分割・評価関数・ハイパラ探索の予算をすべての手法で共有する
- 自明な基準線 (Random / MostPopular) を必ず置く
- 手法は外部ライブラリに任せず実装する

という方針で、近傍法から順に積み上げる。

## 評価プロトコル

- **分割**: ユーザーごとの leave-one-out。最新 1 件を test、その 1 つ前を valid、
  残りを train とする。ML-1M は 77% の評価が「同一秒に入力された塊」の一部なので、
  `(timestamp, item_id)` で安定ソートして分割を決定的にしている。
- **評価**: 全アイテムランキング。履歴に含まれるアイテムは候補から外す。
  アイテムが 3706 件しかないため近似近傍探索は使わない (近似誤差が手法差に混ざる)。
- **同点**: 平均順位で扱う。MostPopular のように未観測アイテムが同点になる手法が
  あり、同点の扱いで有利不利が生まれるため。
- **指標**: HR@K / NDCG@K / MRR。正解が 1 件なので HR@K = Recall@K。
  論文値との比較用に 99 負例サンプリング版の HR@10 / NDCG@10 も併記する
  (サンプリング指標は真の順位と一致しないため、全アイテムランキングの代替にはしない)。
- **暗黙化**: `--min-rating` を指定しなければ全評価を正例とする (NCF 系の流儀)。
  `--min-rating 3` は既存の `../02.KnowledgeGraph_Recommendation` との比較用。

## 実装済みの手法

| 段階 | 手法 | 実装 |
|---|---|---|
| 基準線 | Random / MostPopular | `models/baseline.py` |
| 1. 近傍法 | ItemKNN / UserKNN (asymmetric cosine + shrink + top_k) | `models/knn.py` |
| 2. 行列分解 | implicit ALS (Hu et al. 2008) / BPR-MF (Rendle et al. 2009) | `models/mf.py` |
| 3. 線形 | EASE (Steck 2019) | `models/linear.py` |

## 使い方

```bash
uv venv --python 3.12
uv pip install -r requirements.txt

# 単発実行
PYTHONPATH=src .venv/bin/python -m recommend_ml.run --models most_popular,ease --min-rating 3

# valid でのグリッドサーチ -> 最良設定だけ test で評価
PYTHONPATH=src .venv/bin/python -m recommend_ml.tune --model item_knn \
    --grid '{"top_k":[5,10,50],"shrink":[0,100],"alpha":[0.25,0.5]}'

# 結果の比較表
PYTHONPATH=src .venv/bin/python -m recommend_ml.report --min-rating 3
```

## データ

`data/ml-1m/ratings.dat` を起点にする。データはリポジトリに含めない
(ML-1M の利用条件が再配布を許可していないため)。取得は次のコマンド。

```bash
./scripts/download_data.sh
```

## ディレクトリ

```text
src/recommend_ml/
  data.py       読み込みと leave-one-out 分割
  evaluate.py   順位計算と指標
  models/       手法の実装
  run.py        単発実行
  tune.py       グリッドサーチ
  report.py     結果の集計
results/        結果 CSV とログ
```
