---
title: ロジスティック回帰で与信審査AIを作ってみよう
tags:
  - 機械学習
  - 金融
  - XAI
private: false
updated_at: '2026-03-31T20:54:42+09:00'
id: 2b0f48f3157bc305b099
organization_url_name: null
slide: false
ignorePublish: false
---

# 注意書き

この記事は,著者が個人開発の中で調べたことや試したことを整理した備忘録として書いています.  
あわせて,技術記事という性格上,実際に手元で試す場合は原典や実装,前提条件を確認したうえで各自の責任で扱ってください.

# はじめに
分類モデルの解釈性についていろいろ試してみたく,題材を探していたところ,与信審査AIが面白そうだったため,実際に構築してみようと思います.

# 与信審査AI

いま,金融領域では審査業務の効率化に向けてAIの活用が進んでいます.代表例が, **顧客の過去の融資情報や口座情報をもとに貸し倒れリスクを予測するモデル** です.

従来人間が行っていた審査の一部あるいは全部を自動化,定量化することで, **審査対象が多くなった場合の人手不足の解消** や **審査にかかるナレッジの体系化** などが期待されます.

実務の具体例として,みずほ銀行における与信審査AIプロジェクトについてわかりやすく書かれたインタビューがあったのでぜひ読んでみてください.

https://www.saison-technology.com/column/hulft_square_case_20/
 
# 問題設定
ここからは実データによる分析に入ります.
今回は与信審査AIプロジェクトにおいて以下のような課題を解決したいケースを想定します.
> A社では顧客増加による人手不足を解決するため,申込者が一定期間に深刻な延滞や貸し倒れを発生させるリスクの判定にAIを導入することを検討している.

# データ
アメリカのアナリティクス企業FICOが公開しているHELOC（Home Equity Line of Credit）の与信審査データを使います.目的変数は一定期間内に深刻な延滞がある場合を$ 1 $,ない場合を$ 0 $とラベリングしています.

HELOCデータについての主な情報をまとめました.

|                                    |       | 
| :--------------------------------: | :---: | 
| データ数                           | 10459 | 
| 特徴量数（列数）                      | 24   | 
| 陽性率（ラベルが1のデータの割合） | 約52% | 


全特徴量の定義や型などは以下のページを参照してください.

https://huggingface.co/datasets/mstz/heloc

# 使用モデル
今回は分類モデルとしてロジスティック回帰を用います.ロジスティック回帰は一般的な分類モデルであり,出力は以下の数式で表されます.

```math
\Pr(Y_t = 1)
=
\frac{
e^{\alpha + \beta_1 x_{t1} + \beta_2 x_{t2} + \cdots + \beta_p x_{tp}}
}{
1 + e^{\alpha + \beta_1 x_{t1} + \beta_2 x_{t2} + \cdots + \beta_p x_{tp}}
}
```
細かいモデルの説明はここでは省きますが,以下の点を押さえておくとこの先が読みやすくなります.

- 入力$ x $を受け取り, **そのデータが 1 のクラスに属する確率$ p \in [0,1] $を出す** モデル
- 最終的な出力ラベルは **確率$ p $にしきい値$ \theta $を設けて $ 0 $ / $ 1 $ に変換する** ことで得られる.
- 各特徴量の係数は判定への影響方向を表し, **正の係数は$ 1 $方向,負の係数は$ 0 $方向に働く** と解釈できる.


ロジスティック回帰について,具体例付きのわかりやすい説明が以下のページにあります.

https://gmo-research.ai/research-column/logistic-regression-analysis

数理的な説明はこちらのページが丁寧です.

https://zero2one.jp/learningblog/math-logistic-regression/?srsltid=AfmBOopWeiqWENXEPEPKW0aUZ8HplhNgH2eO2p9j_N6_JdkRspJx8aoY

# データ前処理
実際のコードを記載します.容量の都合でデータ整形以外は割愛するので,コード全体は記事最下部のGitHubを参照してください.
## 特殊値の処理
HELOCデータにはところどころ欠損を表す特殊値が含まれますが,値によって意味が違います.
| 異常値 | 意味                                                       | 
| :----: | :--------------------------------------------------------: | 
| -7     | その項目に対応する事象がそもそも起きていない               | 
| -8     | 取引履歴や照会履歴はある前提でも,使える・有効な記録がない | 
| -9     | 信用情報機関の記録がない,または調査自体が行われていない   | 

今回はすべての列が異常値になっているデータを削除したうえで,残りの欠損を中央値で補完し,それぞれの行に特殊値があったかどうかのフラグ列を用意します.

```python
special_value_unique = {-9, -8, -7}

train_df_transformed = train_df.copy()
test_df_transformed = test_df.copy()

train_all_special_mask = train_df_transformed[feature_columns].isin(special_value_unique).all(axis=1)
test_all_special_mask = test_df_transformed[feature_columns].isin(special_value_unique).all(axis=1)

# 補完の前に,全特徴量が特殊値の行を除外する.
train_df_transformed = train_df_transformed.loc[~train_all_special_mask].reset_index(drop=True)
test_df_transformed = test_df_transformed.loc[~test_all_special_mask].reset_index(drop=True)

# 行除外後のデータに対して,特殊値フラグ列を作成する.
special_value_flag_cols = []
for col in base_feature_columns:
    for special_value in sorted(special_value_unique):
        flag_col = f"{col}_is_special_{abs(special_value)}"
        train_df_transformed[flag_col] = (train_df_transformed[col] == special_value).astype(int)
        test_df_transformed[flag_col] = (test_df_transformed[col] == special_value).astype(int)
        if train_df_transformed[flag_col].sum() > 0 or test_df_transformed[flag_col].sum() > 0:
            special_value_flag_cols.append(flag_col)

# -7 は 0 で補完する.
train_df_transformed.loc[:, base_feature_columns] = train_df_transformed[base_feature_columns].replace(-7, 0)
test_df_transformed.loc[:, base_feature_columns] = test_df_transformed[base_feature_columns].replace(-7, 0)

# -8 と -9 は train の中央値で補完する.
train_special_mask = train_df_transformed[base_feature_columns].isin({-9, -8})
test_special_mask = test_df_transformed[base_feature_columns].isin({-9, -8})
train_df_transformed[base_feature_columns] = train_df_transformed[base_feature_columns].mask(train_special_mask, np.nan)
test_df_transformed[base_feature_columns] = test_df_transformed[base_feature_columns].mask(test_special_mask, np.nan)

median_values = train_df_transformed[base_feature_columns].median()
train_df_transformed[base_feature_columns] = train_df_transformed[base_feature_columns].fillna(median_values)
test_df_transformed[base_feature_columns] = test_df_transformed[base_feature_columns].fillna(median_values)

```

## 特徴量作成
既存特徴量から新しい特徴量を作ります.今回は以下のような特徴量を追加しました.
| 特徴量名                         | 意味                                                                                                     | 
| :------------------------------: | :------------------------------------------------------------------------------------------------------: | 
| credit_history_length            | 初回取引からの経過月数                                                                                   | 
| recent_activity_gap              | 初回取引からの経過月数と直近取引からの経過月数の差,最近どれだけ取引活動があったか                       | 
| delinquency_trade_sum            | 60日超延滞件数と90日超延滞件数の合計                                                                     | 
| delinquency_trade_ratio          |  重い延滞件数合計を総取引件数で割った比率,取引全体に占める延滞の多さ                                    | 
| satisfactory_trade_ratio         | 良好に完了した取引件数を総取引件数で割った比率,健全な取引の比率                                         | 
| balance_trade_ratio              |  残高ありのリボルビング取引件数と分割払い取引件数の合計を総取引件数で割った比率                          | 
| revolving_installment_burden_gap | リボルビング型の負担率と分割払い型の負担率の差                                                           | 
| inquiry_pressure                 |  過去6か月の照会件数を総取引件数で割った比率,取引規模に対して最近の照会がどれだけ多いか                 | 
| recent_inquiry_share             | 「ごく最近を除く過去6か月の照会件数」を,過去6か月の総照会件数で割った比率.直近照会に偏っているかどうか | 
| illegal_trade_gap                | 全期間での問題取引最大件数と過去1年での問題取引最大件数の差.問題取引が最近よりも過去に多かったか        | 
| high_ratio_bank_share            | 高い利用率を示す金融機関数を総取引件数で割った比率,取引規模に対して高負担先がどれだけあるか             | 

```python
def add_credit_risk_features(input_df: pd.DataFrame) -> pd.DataFrame:
    df_feat = input_df.copy()

    total_trades = df_feat["nr_total_trades"].replace(0, np.nan)
    revolving_balance_trades = df_feat["nr_revolving_trades_with_balance"].clip(lower=0)
    installment_balance_trades = df_feat["nr_installment_trades_with_balance"].clip(lower=0)
    delinquency_60 = df_feat["nr_trades_insolvent_for_over_60_days"].clip(lower=0)
    delinquency_90 = df_feat["nr_trades_insolvent_for_over_90_days"].clip(lower=0)
    inquiries_6m = df_feat["nr_inquiries_in_last_6_months"].clip(lower=0)

    df_feat["credit_history_length"] = df_feat["months_since_first_trade"].clip(lower=0)
    df_feat["recent_activity_gap"] = (
        df_feat["months_since_first_trade"].clip(lower=0)
        - df_feat["months_since_last_trade"].clip(lower=0)
    ).clip(lower=0)
    df_feat["delinquency_trade_sum"] = delinquency_60 + delinquency_90
    df_feat["delinquency_trade_ratio"] = (df_feat["delinquency_trade_sum"] / total_trades).fillna(0)
    df_feat["satisfactory_trade_ratio"] = (
        df_feat["number_of_satisfactory_trades"].clip(lower=0) / total_trades
    ).fillna(0)
    df_feat["balance_trade_ratio"] = (
        (revolving_balance_trades + installment_balance_trades) / total_trades
    ).fillna(0)
    df_feat["revolving_installment_burden_gap"] = (
        df_feat["net_fraction_of_revolving_burden"] - df_feat["net_fraction_of_installment_burden"]
    )
    df_feat["inquiry_pressure"] = (inquiries_6m / total_trades).fillna(0)
    df_feat["recent_inquiry_share"] = (
        df_feat["nr_inquiries_in_last_6_months_not_recent"].clip(lower=0)
        / inquiries_6m.replace(0, np.nan)
    ).fillna(0)
    df_feat["illegal_trade_gap"] = (
        df_feat["maximum_illegal_trades"] - df_feat["maximum_illegal_trades_over_last_year"]
    )
    df_feat["high_ratio_bank_share"] = (
        df_feat["nr_banks_with_high_ratio"].clip(lower=0) / total_trades
    ).fillna(0)

    return df_feat

```

## スケーリング

分布が極端な列について,対数変換します.[^scaling]
```python
#歪度（skewness）を確認してlogスケーリング候補列を抽出
skew_threshold = 1.0

skew_base_columns = [
    col for col in num_columns
    if col in train_df_transformed.columns and col != target_col
]

skew_summary = (
    train_df_transformed[skew_base_columns]
    .agg(["skew", "min", "max"])
    .T.rename(columns={"skew": "skewness", "min": "min_value", "max": "max_value"})
    .sort_values("skewness", key=lambda s: s.abs(), ascending=False)
)
skew_summary["abs_skewness"] = skew_summary["skewness"].abs()
skew_summary["recommended_transform"] = np.select(
    [
        (skew_summary["skewness"] >= skew_threshold) & (skew_summary["min_value"] >= 0),
        skew_summary["abs_skewness"] >= skew_threshold,
    ],
    ["log1p", "consider_other_transform"],
    default="keep_as_is",
)

skewed_cols = skew_summary.query("abs_skewness >= @skew_threshold").index.tolist()
log_candidate_cols = skew_summary.query(
    "skewness >= @skew_threshold and min_value >= 0"
).index.tolist()

skew_summary.head(30)
```

```python
#変換
available_log_cols = [col for col in log_candidate_cols if col in train_df_transformed.columns]

for transformed_df in [train_df_transformed, test_df_transformed]:
    for col in available_log_cols:
        transformed_df[f"{col}_log1p"] = np.log1p(transformed_df[col].clip(lower=0))

log_output_columns = [f"{col}_log1p" for col in available_log_cols]
num_columns = sorted(set(num_columns + log_output_columns) - set(available_log_cols))
drop_columns = sorted(set(drop_columns + available_log_cols))
```

# モデル作成・予測
全データのうち8割を学習データ,2割をテストデータとして使用し,モデルを構築します.まず準備として,モデルが出力する確率を$ 0/1 $に変換するための閾値$ \theta $を最適化します.

今回は,訓練データを用いた交差検証[^cv]により平均のF1スコア[^metrics]が最も高い値を用います.Recallを重視したい一方で,Precisionとのバランスも見たいため,閾値はF1を基準に決めます.

```python
threshold_grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
cv_metric_rows = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_full, y_train_full), start=1):
    X_train_fold = X_train_full.iloc[train_idx]
    X_valid_fold = X_train_full.iloc[valid_idx]
    y_train_fold = y_train_full[train_idx]
    y_valid_fold = y_train_full[valid_idx]

    fold_scaler = StandardScaler()
    X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
    X_valid_fold_scaled = fold_scaler.transform(X_valid_fold)

    fold_model = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    )
    fold_model.fit(X_train_fold_scaled, y_train_fold)

    fold_valid_pred_proba = fold_model.predict_proba(X_valid_fold_scaled)[:, 1]

    for threshold in threshold_grid:
        fold_valid_pred = (fold_valid_pred_proba >= threshold).astype(int)
        cv_metric_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "accuracy": accuracy_score(y_valid_fold, fold_valid_pred),
                "precision": precision_score(y_valid_fold, fold_valid_pred, zero_division=0),
                "recall": recall_score(y_valid_fold, fold_valid_pred, zero_division=0),
                "f1": f1_score(y_valid_fold, fold_valid_pred, zero_division=0),
            }
        )

cv_metrics_df = pd.DataFrame(cv_metric_rows)
logistic_threshold_candidates = (
    cv_metrics_df.groupby("threshold", as_index=False)
    .agg(
        accuracy=("accuracy", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    .sort_values(["f1", "threshold"], ascending=[False, True])
    .reset_index(drop=True)
)

logistic_best_threshold = (
    logistic_threshold_candidates.iloc[0]["threshold"]
    if not logistic_threshold_candidates.empty else 0.5
)

print("cross-validation completed")
print(f"selected threshold from 5-fold mean f1: {logistic_best_threshold:.4f}")


```
```python
# 先にデータ全体を標準化する
scaler = StandardScaler()

X_train_full_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full),
    columns=feature_cols,
    index=X_train_full.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols,
    index=X_test.index,
)

#学習
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=random_state,
)
logistic_model.fit(X_train_full_scaled, y_train_full)

#予測
logistic_test_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]
logistic_test_pred = (logistic_test_pred_proba >= logistic_best_threshold).astype(int)
```

# 評価
分類モデルの評価指標[^metrics]は様々あります.代表的なものを以下に示します.
| 指標 | 和訳 | 意味 |
|---|---|---|
| Accuracy | 正解率 | 全体の予測のうち,正しく予測できた割合.クラス不均衡が大きい場合は高く見えやすい. |
| Precision | 適合率 | 陽性と予測したもののうち,実際に陽性だった割合.偽陽性をどれだけ抑えられているかを見る. |
| Recall | 再現率 | 実際に陽性であるもののうち,陽性と正しく予測できた割合.偽陰性をどれだけ減らせているかを見る. |
| F1-score | --- | 適合率と再現率の調和平均.両者のバランスを評価するための指標. |

これらの指標はある程度互いにトレードオフの関係にあるため,プロジェクトに合わせて正しい評価指標を設計することが肝要です.

与信審査においては「貸し倒れリスクの回避」が主目的であるため,なるべくラベル1のデータを見逃したくありません.そのため今回はRecall（再現率）を重視します.

```python
logistic_metrics = {
    "accuracy": accuracy_score(y_test, logistic_test_pred),
    "precision": precision_score(y_test, logistic_test_pred, zero_division=0),
    "recall": recall_score(y_test, logistic_test_pred, zero_division=0),
    "f1": f1_score(y_test, logistic_test_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, logistic_test_pred_proba),
}

logistic_metrics_df = pd.DataFrame(logistic_metrics, index=["test"]).T.rename(columns={"test": "value"})
display(logistic_metrics_df)
```
実際の指標を以下に示します.recallが0.83なので,実際にリスクありと判定すべき人のうち8割以上を検出できたということになりますね.

| 評価指標  | 値   | 
| :-------: | :--: | 
| accuracy  | 0.73 | 
| precision | 0.71 | 
| recall    | 0.83 | 
| f1        | 0.77 | 
| roc_auc   | 0.80 | 

余談ですが,理論上すべてを$ 1 $と判定しておけばrecallは$ 1.00 $です.そのため実際はrecall以外の指標が極端に悪くないかも気にする必要があります.

# まとめ
本記事ではロジスティック回帰を用いて与信審査AIを作成,評価していきました.評価指標を見る限りはそれなりに汎化性能の高いモデルが作成できたと思います.次回はこのモデルを足掛かりにモデルを解釈する手法を学んでいきます.

# 参考文献
- [The Regression Analysis of Binary Sequences](https://academic.oup.com/jrsssb/article/20/2/215/7027376)
- [Application of the Logistic Function to Bio-Assay](https://www.jstor.org/stable/2280041?origin=crossref)

# Githubリンク
後日追記予定



[^scaling]:分布が右に歪んでいる場合,対数変換をかけて分布の偏りを和らげることがあります.[こちら](https://qiita.com/tk-tatsuro/items/86c0a9ff744f73ad1832#6%E5%AF%BE%E6%95%B0%E5%A4%89%E6%8F%9B)の記事で分かりやすく解説されています.
[^cv]:交差検証は学習データを複数に分け,学習と評価を入れ替えながら繰り返すことでモデルの汎化性能を安定して評価する方法です.[こちら](https://qiita.com/Hatomugi/items/620c1bc757266b00e87f)の記事で分かりやすく解説されています.
[^metrics]:分類モデルの評価指標について,[こちら](https://qiita.com/keiji_dl/items/0a8130aea8233fca92e0)の記事で分かりやすく解説されています.
