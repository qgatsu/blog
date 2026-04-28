---
title: 効果検証入門① 効果検証の練習用データを生成してみる
tags:
  - 機械学習
  - 因果推論
  - 効果検証
private: false
updated_at: '2026-04-17T16:54:37+09:00'
id: 2b38b8262c5ce635535b
organization_url_name: null
slide: false
ignorePublish: false
---

# 注意書き

この記事は,著者が個人開発の中で調べたことや試したことを整理した備忘録として書いています.  
あわせて技術記事という性格上,実際に手元で試す場合は原典や実装,前提条件を確認したうえで各自の責任で扱ってください.

# はじめに

効果検証を勉強したいんですが,シンプルかつ手元ですぐ使える公開データはそこまで多くありません.
というわけでまずは自分でサンプルデータを作ります.今回は **クーポン配布の効果検証** を想定して,コスメECっぽい顧客データをシンプルに生成します.

# データ概要

今回は1行を1顧客として以下のような列を持つデータを作ります.

| 列名 | 意味 |
| --- | --- |
| `id` | 顧客ID |
| `gender` | 性別 |
| `age` | 年齢 |
| `recency` | 直近購買からの日数 |
| `frequency` | 購買回数 |
| `amount` | 通常時の購買金額 |
| `tau` | クーポンによる真の上乗せ効果 |
| `treatment_rate` | クーポン配布確率 |
| `treatment` | 実際に配布されたかどうか |
| `y0` | 非配布時の潜在アウトカム |
| `y1` | 配布時の潜在アウトカム |
| `outcome` | 実際に観測された購買金額 |

コスメEC想定なので性別はざっくり **男性2 : 女性8** くらいの分布にしています.全体としては,できるだけシンプルにしつつ`treatment_rate` と `tau` が観測特徴に依存するようにして,効果検証の練習に使いやすい形を目指しています.

# 実装

## パラメータ設定

```python
SEED = 42
N = 50_000

# 性別カテゴリとその出現比率
# コスメECを想定して,男性2 : 女性8くらいにする
GENDER_LABELS = ["m", "f"]
GENDER_PROBS = [0.2, 0.8]

# 年齢分布
AGE_MEAN = 30
AGE_STD = 10
AGE_MIN = 18
AGE_MAX = 70

# recency の分布
# 直近購買からの日数をざっくり正規分布で置く
RECENCY_MEAN = 10
RECENCY_STD = 6
RECENCY_MIN = 0
RECENCY_MAX = 30

# frequency の分布
# 購買回数なのでポアソン分布を使う
FREQUENCY_MEAN = 5
FREQUENCY_LAMBDA = 5

# amount の生成パラメータ
# beta0: 金額のベース水準
# beta1: 購買頻度が高い人ほど金額が上がる度合い
# beta2, amount_age_peak: 年齢によるなだらかな山型を作るための係数
beta0 = 4000
beta1 = 3500
beta2 = 180
amount_age_peak = 40

# amount に載せるノイズ
amount_noise_mean = 0
amount_noise_std = 900

# treatment_rate の生成パラメータ
# 若めの層にクーポンを出しやすい,という想定を置く
TREATMENT_AGE_MEAN = 20
TREATMENT_AGE_STD = 8
TREATMENT_AMOUNT_STD = 3000
TREATMENT_MAX = 1.0
TREATMENT_RATE_SCALE = 0.75
TREATMENT_NOISE_MEAN = 0
TREATMENT_NOISE_STD = 20
TREATMENT_NOISE_SCALE = 1000

# tau の生成パラメータ
# クーポン効果は30代半ばかつ低-中価格帯で少し出やすい形にする
TAU_AGE_MEAN = 35
TAU_AGE_STD = 9
TAU_AMOUNT_STD = 3000
TAU_MAX = 2000

# 観測アウトカムに載せるノイズ
OUTCOME_NOISE_MEAN = 0
OUTCOME_NOISE_STD = 500

rng = np.random.default_rng(SEED)
```

## 顧客属性の生成

まずは `id`, `gender`, `age`, `recency`, `frequency`, `amount` を作ります.  
`amount` は頻度が高いと増えやすく年齢にはゆるい山を持たせています.

`amount` の設計は次の式に対応しています.

```math
\mathrm{amount}_i
=
\beta_0
+ \beta_1 \sqrt{\mathrm{frequency}_i}
- \beta_2 \frac{(\mathrm{age}_i - \mathrm{amount\_age\_peak})^2}{100}
+ \varepsilon_i
```

ここで $ \varepsilon_i $ は平均 `0`,標準偏差 `amount_noise_std` のノイズです.  
頻度が高いほど購買金額は増えやすく,年齢についてはピーク年齢の周辺でやや高くなるようにしています.

```python
amount_noise = np.rint(
    rng.normal(loc=amount_noise_mean, scale=amount_noise_std, size=N)
).astype(int)

user_df = pd.DataFrame(
    {
        "id": np.arange(1, N + 1),
        "gender": rng.choice(GENDER_LABELS, size=N, p=GENDER_PROBS),
    }
)

user_df["age"] = np.clip(
    np.rint(rng.normal(loc=AGE_MEAN, scale=AGE_STD, size=N)),
    AGE_MIN,
    AGE_MAX,
).astype(int)

user_df["recency"] = np.clip(
    np.rint(rng.normal(loc=RECENCY_MEAN, scale=RECENCY_STD, size=N)),
    RECENCY_MIN,
    RECENCY_MAX,
).astype(int)

user_df["frequency"] = rng.poisson(lam=FREQUENCY_LAMBDA, size=N).astype(int)

user_df["amount"] = (
    beta0
    + beta1 * np.sqrt(user_df["frequency"])
    - beta2 * ((user_df["age"] - amount_age_peak) ** 2) / 100
    + amount_noise
).astype(int)

```

## 介入率と介入効果の生成

次に,クーポン配布確率 `treatment_rate` と真の介入効果 `tau` を作ります.  
どちらも `age` と `amount` に依存する分布から作っています.

まず `tau` です.

```python
amount_mean = user_df["amount"].mean()
amount_q25 = user_df["amount"].quantile(0.25)

tau_age_score = np.exp(-0.5 * ((user_df["age"] - TAU_AGE_MEAN) / TAU_AGE_STD) ** 2)
tau_amount_score = np.exp(-0.5 * ((user_df["amount"] - amount_q25) / TAU_AMOUNT_STD) ** 2)
tau_base = tau_age_score * tau_amount_score
tau_base = tau_base / tau_base.max()
user_df["tau"] = np.clip(
    TAU_MAX * tau_base,
    0,
    TAU_MAX,
).astype(int)

```

続いて `treatment_rate` です.

```python
treatment_age_score = np.exp(-0.5 * ((user_df["age"] - TREATMENT_AGE_MEAN) / TREATMENT_AGE_STD) ** 2)
treatment_amount_score = np.exp(-0.5 * ((user_df["amount"] - amount_mean) / TREATMENT_AMOUNT_STD) ** 2)
treatment_base = treatment_age_score * treatment_amount_score
treatment_base = treatment_base / treatment_base.max()
treatment_noise = np.rint(
    rng.normal(loc=TREATMENT_NOISE_MEAN, scale=TREATMENT_NOISE_STD, size=N)
).astype(int)

user_df["treatment_rate"] = np.clip(
    TREATMENT_RATE_SCALE * TREATMENT_MAX * treatment_base + treatment_noise / TREATMENT_NOISE_SCALE,
    0,
    TREATMENT_MAX,
)

```

最後に,その配布確率に従って `treatment` をサンプリングします.

```python
user_df["treatment"] = rng.binomial(n=1, p=user_df["treatment_rate"]).astype(int)

treatment_count_df = (
    user_df["treatment"]
    .value_counts()
    .sort_index()
    .rename_axis("treatment")
    .reset_index(name="count")
)

```

## 観測アウトカムの生成

非配布時の潜在アウトカムを `y0`, 配布時の潜在アウトカムを `y1` として,実際に観測される `outcome` を作ります.

```python
user_df["y0"] = user_df["amount"].astype(int)
user_df["y1"] = (user_df["y0"] + user_df["tau"]).astype(int)

outcome_noise = np.rint(
    rng.normal(loc=OUTCOME_NOISE_MEAN, scale=OUTCOME_NOISE_STD, size=N)
).astype(int)

user_df["outcome"] = np.where(
    user_df["treatment"] == 1,
    user_df["y1"],
    user_df["y0"],
)
user_df["outcome"] = np.clip(user_df["outcome"] + outcome_noise, 0, None).astype(int)

```

## 分布の確認

`treatment_rate` と `tau` は,どちらも `age` と `amount` に依存した山型の分布になるように作っています.

![tau_treatment_distribution.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/93490ab4-56f8-48dd-9f78-cd8db405f5fe.png)


# データ配布

以下の[Drive](https://drive.google.com/drive/folders/18tQwuuLQ488XeztiMTQY7MHfOak_NCk-?usp=sharing)から　EC_coupon_data.zip　をダウンロード可能です。

# まとめ

今回は効果検証の練習に使える最小限のデータを生成しました.  
顧客属性,配布確率,介入効果,観測アウトカムまでを一通りそろえておくことで手元ですぐに推定や可視化を試せる状態にしています.  

次回以降はこのデータを使って実際に効果検証を試していきます.
