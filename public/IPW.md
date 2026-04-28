---
title: 効果検証入門② IPWでクーポン施策を見てみる
tags:
  - 機械学習
  - 因果推論
  - 効果検証
private: false
updated_at: '2026-04-17T16:54:41+09:00'
id: 7788c7e44c2027d74aa9
organization_url_name: null
slide: false
ignorePublish: false
---

# 注意書き

この記事は,著者が個人開発の中で調べたことや試したことを整理した備忘録として書いています.  
あわせて,技術記事という性格上実際に手元で試す場合は原典や実装,前提条件を確認したうえで各自の責任で扱ってください.

# はじめに

効果検証や因果推論を勉強し始めると,かなり早い段階で **「介入群と非介入群の平均をそのまま比べてよいのか」** という壁にぶつかります.

たとえばクーポン施策を考えると,企業は完全にランダムにクーポンを配るとは限りません.年齢や購買頻度,購買金額などを見ながら **「効きそうな人」** に寄せて配ることがあります.このとき介入群と非介入群ではもともとの顧客属性がかなり違ってしまいます.

そこで本記事では,傾向スコアを使った代表的な補正法の1つである IPW（Inverse Probability Weighting, 逆確率重みづけ）を扱います.あわせて,シミュレーションデータを使って

- 真の平均介入効果 `ATE` を確認する
- 単純平均差と IPW を比較する
- 傾向スコアを機械学習で推定する
- データ数を変えたときの挙動を観察する

という流れで見ていきます.

# そもそも効果検証では何がしたいのか

因果推論でやりたいことをかなり素朴に言うと, **「その施策をやったから結果がどう変わったのか」** を知ることです.

個人 $ i $ に対して,介入ありの結果を $ Y_i(1) $,介入なしの結果を $ Y_i(0) $ と書くと[^potential_outcome],本当は

```math
\tau_i = Y_i(1) - Y_i(0)
```

を見たいわけです.ただし現実には,同じ人に対して **同時に両方の世界を観測することはできません**.クーポンを配った世界と配らなかった世界を,同じ顧客について同時に観測することはできないからです.

そこで個票効果そのものではなく,平均として

```math
\mathrm{ATE} = \mathbb{E}[Y(1) - Y(0)]
```

を推定したくなります.

ここで `ATE` は Average Treatment Effect の略で,介入効果を個人ごとではなく母集団全体で平均した量を指します.個々のサンプルでは `Y(1)` と `Y(0)` を同時に観測できませんが,全体で期待値を取ることで **「介入した場合としない場合で平均的にどれだけ差があるか」** を近似的に捉えよう,というのが基本の考え方です.

# データ

今回は以下の記事で作成したクーポン施策のシミュレーションデータを使います.

https://qiita.com/na9atsuki/items/2b38b8262c5ce635535b

データ数は `50,000` 件で,主な列は以下の通りです.

- `age`, `recency`, `frequency`, `amount`: 顧客属性と過去購買行動
- `treatment_rate`: データ生成時に与えた真の介入率
- `treatment`: 実際に介入されたかどうか
- `tau`: 真の個別介入効果
- `y0`, `y1`: 介入なし・ありそれぞれの潜在アウトカム
- `outcome`: 実際に観測されたアウトカム

集計すると,全体の介入率は約 `0.309` ,真の `tau` の平均,すなわち真の `ATE` は `957.74` でした.

| 指標 | 値 |
| --- | ---: |
| データ数 | `50,000` |
| 平均年齢 | `30.54` |
| 平均購買金額| `11,282.92` |
| 介入率  | `0.309` |
| 真のATE | `957.74` |

# 単純平均差で ATE を予測してみる

まずはベースラインとして,介入群と非介入群の観測アウトカム平均の差をそのまま使います.

```math
\hat{\tau}_{\mathrm{naive}} = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]
```

コードはこれだけです.

```python
treated_mean = df.loc[df["treatment"] == 1, "outcome"].mean()
control_mean = df.loc[df["treatment"] == 0, "outcome"].mean()
tau_hat_naive = treated_mean - control_mean

true_ate = df["tau"].mean()
ate_error_naive = abs(true_ate - tau_hat_naive)
```

結果は以下でした.

| 指標 | 値 |
| --- | ---: |
| 真の `ATE` | `957.74` |
| 介入群平均 | `11,978.60` |
| 非介入群平均 | `11,316.39` |
| 単純平均差 | `662.21` |
| 絶対誤差 | `295.52` |

![naive_ate.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/e20b17e9-1e67-4d8b-accd-4e83829196f0.png)

真の `ATE` は `957.74` なのに,単純平均差は `662.21` まで下がっており,かなり過小推定になっています.

このズレは,介入の割当てがランダムではなく顧客属性に応じて偏っているためです.つまり **介入群と非介入群の「元々の違い」** が残ったまま比較しているため施策の純粋な効果だけを取り出せていません.

# 傾向スコア予測

## 傾向スコアとは

傾向スコアは

```math
e(x) = P(T=1 \mid X=x)
```

で表される, **「その属性を持つサンプルが介入される確率」** です.[^rosenbaum_rubin]

これを推定できるとたとえば

- もともと介入されやすい人が介入されているのか
- 逆に,かなり介入されにくいのにたまたま介入された人がいるのか

を確率として扱えるようになります.

IPW は,この確率の逆数で重みづけすることで,観測データから **「もし介入割当てがもっとランダムに近かったら」** という擬似母集団[^pseudo_population]を作ろうとする考え方です.

## 傾向スコアの予測

今回は `treatment` を目的変数にしてXGBoost で傾向スコアを推定します.説明変数には `gender`, `age`, `recency`, `frequency`, `amount` を使い介入後の情報や真の効果そのものは使いません.

また,各行の傾向スコアをその行自身で学習したモデルから出してしまうと楽観的になるため,`StratifiedKFold` を使って `out-of-fold`[^oof] 予測を作ります.

```python
feature_cols = ["gender", "age", "recency", "frequency", "amount"]
target_col = "treatment"

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gender"]),
        ("num", "passthrough", ["age", "recency", "frequency", "amount"]),
    ]
)

propensity_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_propensity_score = np.zeros(len(df))

for train_idx, valid_idx in cv.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    propensity_model.fit(X_train, y_train)
    oof_propensity_score[valid_idx] = propensity_model.predict_proba(X_valid)[:, 1]
```

以下は実際の介入,非介入群ごとの予測介入確率の分布です.

![propensity_score_estimated_dist.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/23dcde15-b5eb-4f1d-a030-93936970aa59.png)

この設定での OOF ROC-AUC[^roc_auc] は `0.8317` でした.完全な予測ではありませんが介入割当ての構造をある程度は捉えられています.

# IPW（逆確率重みづけ法）

## IPWとは

IPW は,観測された各サンプルに対して **「その割当てがどれくらい起きにくかったか」** に応じて重みをつける方法です.

介入されたサンプルには `1 / e(X_i)` ,非介入サンプルには `1 / (1 - e(X_i))` の重みをつけます.すると,

- 介入されやすい人が介入されていた場合は重みが小さくなる
- 介入されにくい人が介入されていた場合は重みが大きくなる

という補正がかかります.

ATE の推定量としては各サンプルの擬似寄与を

```math
\hat{\tau}_i
=
\frac{T_i Y_i}{\hat{e}(X_i)}
-
\frac{(1-T_i)Y_i}{1-\hat{e}(X_i)}
```

とおき,その平均で

```math
\hat{\tau}
=
\frac{1}{n}\sum_{i=1}^n \hat{\tau}_i
```

を計算しています.

「逆重みづけ」という名前はまさにこの **介入確率の逆数で重みをつける** ところから来ています.

また,分母に入る $ \hat{e}(X_i) $ や $ 1-\hat{e}(X_i) $ が `0` に近づくほど割り算の値が急に大きくなり,一部のサンプルだけが全体の推定を強く引っ張ってしまいます.そのためここでは傾向スコアの極端な値を少しだけ丸めてスコアが暴れすぎないようにしています.

```python
eps = 1e-3
df_ipw = df_ps.copy()
df_ipw["propensity_score_clipped"] = df_ipw["propensity_score_est"].clip(eps, 1 - eps)

df_ipw["tau_hat"] = (
    df_ipw["treatment"] * df_ipw["outcome"] / df_ipw["propensity_score_clipped"]
    - (1 - df_ipw["treatment"]) * df_ipw["outcome"] / (1 - df_ipw["propensity_score_clipped"])
)

ate_ipw = df_ipw["tau_hat"].mean()
```

## IPWの実装結果

結果は次の通りでした.

| 指標 | 値 |
| --- | ---: |
| 真の `ATE` | `957.74` |
| IPW による推定値 | `1,031.58` |
| 絶対誤差 | `73.84` |

単純平均差の絶対誤差が `295.52` だったので,IPW によってかなり改善しています.

![IPW_ate.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/35b66fb3-cbc0-49bb-b4e4-c5b80a26e6e6.png)

比較するとこうなります.

| 手法 | ATE 推定値 | 真値との絶対誤差 |
| --- | ---: | ---: |
| 単純平均差 | `662.21` | `295.52` |
| IPW + XGBoost | `1,031.58` | `73.84` |

少なくとも **傾向スコアを使った補正を入れるだけで単純平均差より真値にかなり近づく** ことがわかります.

## IPWの解釈上の注意

IPW は便利ですがいつでも安定とは限りません.特に重要なのは次の2点です.

- 傾向スコアが極端だと重みが非常に大きくなり,推定量の分散が大きくなりやすい
- 傾向スコアモデルがずれていると,補正の土台自体が崩れる

今回も個票レベルの `tau_hat` を見るとかなり大きな値が出ており,平均すれば使える一方で,サンプル単位ではかなりばらついています.このあたりが,IPW の扱いに慎重さが必要な理由です.

# データ数を変えた場合の IPW の挙動

最後に,介入率を `3` 割に固定したままデータ数だけを変え,ATE 推定誤差がどう変わるかを確認しました.対象のサンプルサイズは

```python
sample_sizes = [100, 1000, 5000, 10000, 50000]
```

です.

結果は以下でした.

| サンプル数 | 単純平均差 MAE | IPW MAE |
| --- | ---: | ---: |
| `100` | `507.77` | `31,843.18` |
| `1,000` | `611.30` | `2,006.76` |
| `5,000` | `299.76` | `859.61` |
| `10,000` | `372.22` | `446.50` |
| `50,000` | `277.99` | `64.47` |

![IPW_datanum.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/7826c00a-16f5-4e9f-82ed-f729c9b2ccc3.png)

かなりはっきりしているのは,**データ数が少ないと IPW は不安定になりやすい** ということです.とくに `100` 件や `1,000` 件では傾向スコア推定の誤差と極端重みの影響が強く出て,単純平均差より悪化しています.

一方で `50,000` 件まで増やすと IPW の MAE は `64.47` まで下がりました.この結果だけを見ると,

- 小標本では IPW はかなり荒れやすい
- サンプルが十分あると真値へ近づきやすい

という挙動が確認できます.

つまり IPW は **「理屈として補正できる」ことと「有限サンプルで安定して使える」ことが別** である点に注意が必要です.

# まとめ

本記事ではクーポン施策のシミュレーションデータを使って,単純平均差と IPW を比較しました.

単純平均差は実装が簡単ですが,介入群と非介入群の属性分布がずれていると大きくバイアスします.一方で IPW は,傾向スコアの逆数で重みづけすることで割当ての偏りを補正し,今回の `50,000` 件データでは真の `ATE = 957.74` に対して `1,031.58` まで近づきました.

一方でデータ数を変化させた実験により,小標本では極端重みによってかなり不安定になることも確認できました.そのため, **傾向スコアの妥当性を確認すること** と **重みの暴れ方を必ず点検すること** が重要そうです.

[^rosenbaum_rubin]: 傾向スコアの古典的な定義は Rosenbaum, Paul R., and Donald B. Rubin. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 70.1 (1983): 41-55. に基づきます.
[^potential_outcome]: 因果推論では, $ Y_i(1) $ を「介入ありの世界でのアウトカム」, $ Y_i(0) $ を「介入なしの世界でのアウトカム」として書き分けます.
[^pseudo_population]: 擬似母集団とは,重みづけによって介入群と非介入群の属性分布がより近づいたとみなせる仮想的な集団のことです.
[^oof]: out-of-fold 予測とは,その行自身を学習に使っていないモデルで予測値を作る方法です.過度に楽観的な予測を避けやすくなります.
[^roc_auc]: ROC-AUC は,しきい値を固定せずに分類モデルの順位付け性能を見る指標です.1 に近いほど,介入群に近いサンプルへ高いスコアを与えられています.

# 参考文献

- Rosenbaum, Paul R., and Donald B. Rubin. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 70.1 (1983): 41-55.
- Hernan, Miguel A., and James M. Robins. *Causal Inference: What If*. Chapman & Hall/CRC, 2020.
