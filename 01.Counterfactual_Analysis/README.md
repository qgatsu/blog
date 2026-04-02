# Counterfactual Analysis for Credit Risk

このプロジェクトは、loan の与信データを使って、与信審査における反実仮想分析を行うものです。

目的は、単に「承認されるか / 否決されるか」を予測することではありません。  
否決になった申込者に対して、

- なぜ否決になったのか
- どの特徴が判定に強く効いていたのか
- どの特徴がどの程度変わっていれば承認側に近づけたのか

を、モデルベースで説明できる状態を作ることを目指しています。

## このプロジェクトで扱う問い

この分析では、主に次の問いに答えることを狙っています。

- 与信審査の可否に強く影響する特徴は何か
- 否決サンプルを承認側に反転させるには、どの特徴の改善が必要か
- その変化は実務的に解釈しやすいか
- 反実仮想として提示された変更案は現実的な介入候補になりうるか

## 全体の流れ

分析は大きく3段階で構成しています。

1. 前処理と特徴量設計
2. 与信予測モデルの構築
3. 反実仮想生成と解釈

それぞれの段階で、後続の分析に必要な中間成果物を保存しています。

## 1. 前処理と特徴量設計

ノートブック: `01.loan_preprocess.ipynb`

このステップでは、元データ `data/data.csv` を読み込み、学習に使える形へ整えます。

主な処理は次の通りです。

- train / test の分割
- 特殊値 `-9`, `-8`, `-7` の整理
- 特殊値フラグ列の作成
- 与信判断に関係しやすい派生特徴量の追加
- 歪みの大きい数値列への `log1p` 変換
- 学習用データの保存

このデータでは、特殊値が単なる欠損ではなく、「履歴がない」「条件に該当しない」といった業務上の意味を持つ可能性があります。  
そのため、値を置換するだけでなく、特殊値だったこと自体をフラグとして残しています。

また、反実仮想分析で扱いやすいように、以下のような中間特徴量も作成しています。

- `credit_history_length`
- `recent_activity_gap`
- `delinquency_trade_sum`
- `delinquency_trade_ratio`
- `satisfactory_trade_ratio`
- `balance_trade_ratio`
- `inquiry_pressure`
- `illegal_trade_gap`
- `high_ratio_bank_share`

前処理後のデータは以下に保存されます。

- `data/df_train.parquet`
- `data/df_test.parquet`

## 2. 与信予測モデルの構築

ノートブック: `02.loan_prediction_model.ipynb`

このステップでは、前処理済みデータを使って与信リスク予測モデルを作成します。

実施している内容は次の通りです。

- 前処理済みデータの読み込み
- 学習用 / 検証用データの分割
- `StandardScaler` によるスケーリング
- XGBoost モデルの学習
- Optuna によるハイパーパラメータ探索
- 検証データ上でのしきい値調整
- テストデータでの評価
- SHAP による特徴量寄与の確認
- 比較用の Logistic Regression モデル作成

この構成にしている理由は、精度だけでなく、後段の反実仮想生成と解釈のしやすさも両立させるためです。

- XGBoost は非線形な関係を拾いやすい
- Logistic Regression は係数ベースで解釈しやすい
- SHAP により、モデルがどの特徴を重視しているか確認できる

学習済みモデルは artifact として保存しています。

- `model/loan_xgb_artifact.joblib`
- `model/loan_logistic_artifact.joblib`

これらの artifact には、モデル本体だけでなく、スケーラーや特徴量列名、しきい値なども含めています。

## 3. 反実仮想生成と解釈

ノートブック: `03.loan_counterfactual_analysis.ipynb`

このステップでは、学習済みモデルに対して DiCE を使い、否決サンプルを承認側へ反転させる反実仮想を生成します。

ここで見たいのは、「どの列をどの程度動かせば判定が変わるか」です。

主な処理は次の通りです。

- 保存済み artifact の読み込み
- モデル入力空間の整備
- 反実仮想探索に使う特徴の選定
- 否決サンプルに対する counterfactual の生成
- 判定反転に必要な変化量の確認

特に重要なのは、すべての列を自由に変更可能にしないことです。  
特殊値フラグのように実務上動かしにくい列は固定し、比較的介入可能と考えやすい特徴に絞って探索する前提を置いています。

## フォルダ構成

```text
01.Counterfactual_Analysis/
|-- 01.loan_preprocess.ipynb
|-- 02.loan_prediction_model.ipynb
|-- 03.loan_counterfactual_analysis.ipynb
|-- README.md
|-- feature_descriptions.txt
|-- model_feature_descriptions.txt
|-- data/
|   |-- data.csv
|   |-- df_train.parquet
|   `-- df_test.parquet
`-- model/
    |-- loan_xgb_artifact.joblib
    `-- loan_logistic_artifact.joblib
```

## ファイルの役割

- `data/data.csv`
  元の与信データです。
- `data/df_train.parquet`, `data/df_test.parquet`
  前処理後の学習用・評価用データです。
- `feature_descriptions.txt`
  元特徴量の意味をまとめた説明ファイルです。
- `model_feature_descriptions.txt`
  モデル入力特徴の説明整理に使う補助ファイルです。
- `model/loan_xgb_artifact.joblib`
  XGBoost ベースの学習済み artifact です。
- `model/loan_logistic_artifact.joblib`
  ロジスティック回帰ベースの学習済み artifact です。

## このプロジェクトの価値

この分析の価値は、モデルの予測結果を説明するだけでなく、意思決定を変えるための具体的な方向性を提示できる点にあります。

例えば、単に

> この顧客はリスクが高い

と出すだけでなく、

> 延滞比率が低く、照会圧力が小さく、残高負担が抑えられていれば承認側に近づいた可能性がある

という形で、より行動可能な示唆へつなげられます。

もちろん、反実仮想がそのまま現実の施策になるとは限りません。  
ただし、モデルがどの方向の変化を「承認に近い」と見ているかを整理することで、与信判断の構造を理解しやすくなります。

## 想定している活用

このプロジェクトは、以下のような用途を想定しています。

- 与信モデルの説明可能性を高める
- 否決理由を特徴量レベルで整理する
- 反実仮想を使って改善シナリオを検討する
- XAI / Counterfactual Analysis の実装例として再利用する

## 補足

このリポジトリは、ブログ化や分析メモとしても利用しやすいよう、前処理、モデル構築、反実仮想生成をノートブック単位で分けています。  
そのため、全体を通して読むことも、特定の段階だけ参照することもできます。
