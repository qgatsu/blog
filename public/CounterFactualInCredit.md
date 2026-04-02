---
title: 反実仮想説明で与信審査AIを解釈してみよう
tags:
  - 機械学習
  - XAI
  - 金融
private: false
updated_at: ''
id: null
organization_url_name: null
slide: false
ignorePublish: false
---

## 注意書き

この記事は、著者が個人開発の中で調べたことや試したことを整理した備忘録として書いています。  
あわせて、技術記事という性格上、実際に手元で試す場合は原典や実装、前提条件を確認したうえで各自の責任で扱ってください。

# はじめに

研究や開発を行う中で、機械学習から予測以外の定性的な知見を導く手法に興味を持ちました。
MLモデルを解釈する手法はたとえば
- 回帰係数
- SHAP
- PFI
- LIME

などがありますが、これらは各特徴の重要度や寄与を定量化するもので　**「その先どうするか」**　はユーザーに委ねられます。
これらの解釈アルゴリズムを意思決定に反映しようと考える場合、結局はモデルの出す結果を見てから頭をひねって施策を練る必要があるため意思決定そのものへの適用には限界があります。

本記事では特徴量重要度を超え、施策に直結し得る解釈手法を扱います。

MicroSoft Reserch社開発の「DiCE」は反実仮想的なサンプルを生成することで、 **目的とした出力を得るための説明変数を直接出力** する手法です。

つまり「どういう要素をもとにその結果が出たのか」だけでなく **「どうすればその結果にならなかったのか」** を直接得ることができるため、その点でほかの手法より実際の意思決定に近い解釈アルゴリズムとなっています。

本記事では、HELOC の与信審査データを題材に、DiCEを使った反実仮想データ生成を通じて、施策決定につながる示唆が得られるかを見ていきます。

# DiCEについて
DiCEは一言で言えば **「AIの予測結果を望ましい方向に変えるためにどの特徴をどれだけ変えればよいかを、できるだけ少なく自然な変更で探す仕組み」** です。
## 入力

1. 学習済みモデル
すでに学習が済んだモデルです。2026/3/21現在で **scikit-learn / TensorFlow / PyTorch で作った分類・回帰モデル** に対応しています。
2. 対象サンプル
分析したい実サンプルです。
3. 望ましい出力
希望する出力です。分類モデルであれば 「0だった出力を1にしたい」 、回帰モデルであれば「$ y $以上の出力にしたい」のような設定ができます。
4. 特徴量の情報
具体的には **「意図的に変動させることが可能な特徴量はどれか」** を渡します。「年齢を5下げる」のような非現実的な施策シナリオを出力させないためのガードです。

## 出力
現在用意されているpythonパッケージの仕様では「目標の予測を満たすように調整された各特徴量の値」を$ k $個返します。

(ex) 与信審査予測モデルについて、「年収、勤続年数、借入額」を変動可能な特徴として渡した場合
| 年収 | 勤続年数 | 借入額 | 
| :--: | :------: | :----: | 
| 500  | 3        | 150    | 
| 550  | 4        | 180    | 
| 600  | 5        | 200    | 

## 数学的捕捉
DiCeは内部で以下の最適化問題を解いています。
```math
C(x)=\arg \min _{c_1, \ldots, c_k} \frac{1}{k} \sum_{i=1}^k \operatorname{yloss}\left(f\left(c_i\right), y\right)+\frac{\lambda_1}{k} \sum_{i=1}^k \operatorname{dist}\left(c_i, x\right)-\lambda_2 \operatorname{dpp} \_\operatorname{diversity}\left(c_1, \ldots, c_k\right)
```

ここで$ \operatorname{yloss}\left(f\left(c_i\right), y\right) $ は **「望ましい出力と化そうサンプルを与えた場合の出力の差」** 、$ \operatorname{dist}\left(c_i, x\right) $ は **「対象の実サンプルとの距離」** 、$ \operatorname{dppdiversity}\left(c_1, \ldots, c_k\right) $は **「各シナリオ同士がどれくらい多様性を持っているか」** を表しています。

そのため、上記最適化を解くことで **より少ない変更で結果を変動させる多様なシナリオ** を生成することができます。
 
# 問題設定
前置きが長くなりましたが、ここからは実データによる分析に入ります。
今回は与信審査AIプロジェクトにおいて以下のような課題を解決したいケースを想定します。
> A社では与信審査AIを運用しているが、いちどリスク有判定になったユーザーに再度の申し込みを促すため「どうすれば承認ラインに達するか」の目安を還元したい。

# データ
アメリカのアナリティクス企業FICOが公開しているHELOC(Home Equity Line of Credit) の与信審査データを使います。目的変数は一定期間内に深刻な延滞がある場合を$ 1 $、ない場合を$ 0 $としています。

また、施策に直結する特徴量については意味を後述しますが、全特徴量の定義や型などは以下のページを参照してください。

https://huggingface.co/datasets/mstz/heloc

# 使用モデル以下
今回は分類モデルとして、以下の記事で作成したロジスティック回帰モデルを用います。ざっくりとしたモデルの説明や参考リンクは以下の記事を参照してください。
> 記事URL

細かいモデルの説明はここでは省きますが、以下の点を押さえておくとこの先が読みやすくなります。

- 入力$ x $を受け取り、 **そのデータが 1 のクラスに属する確率$ p \in [0,1] $を出す** モデル
- 最終的な出力ラベルは **確率$ p $にしきい値$ \theta $を設けて $ 0 $ / $ 1 $ に変換する** ことで得られる。
- 各特徴量の係数は判定への影響方向を表し、 **正の係数は$ 1 $方向、負の係数は$ 0 $方向に働く** と解釈できる。



# 分析
以降は訓練済みのモデルがあることを前提としているため、前処理やモデル作成は前回記事を参照してください。

## まずDiCEを使わずに解釈してみる
比較材料としてDiCEを使わずにどれだけ意思決定が可能かを確かめます。ロジスティック回帰における解釈性の代表は回帰係数です。

前述したように、ロジスティック回帰の回帰係数を見ると「どの特徴量がどちら向きに予測を動かしているか」を見ることができます。今回の予測における回帰係数を可視化してみましょう。
赤は正、青は負です。正の係数はその値が大きいほど `is_at_risk = 1` 側に、負の係数は `0` 側に働きます。

![logreg_coefficient.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/07a37ee8-59d5-41b8-b749-1e1e216d1ce6.png)

絶対値の大きい特徴量を上から見ると次のようになります。

| 特徴名 | 意味 | 係数 | 解釈メモ |
|---|---|---:|---|
| `recent_activity_gap` | 最近の取引活動の厚み | -0.65 | 大きいほど低リスク側 |
| `months_since_last_illegal_trade_is_special_7` | 問題取引が起きていないフラグ | -0.54 | 問題取引なしは低リスク側 |
| `delinquency_trade_sum_log1p` | 延滞関連件数の合計の対数変換 | 0.42 | 延滞が多いほど高リスク側 |
| `estimate_of_risk` | 信用リスク推定スコア | -0.36 | スコアが高いほど低リスク側 |
| `months_since_last_inquiry_not_recent_log1p` | 最後の照会からの経過月数の対数変換 | -0.31 | 最後の照会が遠いほど低リスク側 |

解釈は概ね直感的で妥当に見えます。ただこれだけ見ても実際の施策に結びつけるのは難しいでしょう。例えば今回の場合 **「信用リスク推定スコアを上げよう」や「過去の問題取引をなかったことにしましょう」は変更できないステータスに介入しようとしているため施策としては不適切** です。

このように特徴量重要度はモデルのふるまいを定量的に表現することはできますが、定性的な施策に結びつけるためにはここからもう1つステップが必要です。

## DiCEを使ってみる。
ここからは実際にDiCEを使って分析をしていきます。

### DiCEをインストールする
```python
%pip install dice-ml
```
### DiCEにモデルラッパーとデータラッパーを渡す
データロード後、既定のラッパーでモデルとデータを渡します。

```python
class DiceReadyModel:
    def __init__(self, model, feature_names, threshold):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def _prepare(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df[self.feature_names].copy()

    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(input_df)
        return self.model.predict_proba(X)

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        positive_proba = self.predict_proba(input_df)[:, 1]
        return (positive_proba >= self.threshold).astype(int)


dice_ready_model = DiceReadyModel(
    model=model,
    feature_names=feature_cols,
    threshold=best_threshold,
)

dice_data = dice_ml.Data(
    dataframe=model_train_df,
    continuous_features=feature_cols,
    outcome_name=target_col,
)

dice_model = dice_ml.Model(
    model=dice_ready_model,
    backend="sklearn",
    model_type="classifier",
)

dice = dice_ml.Dice(dice_data, dice_model, method="genetic")
```
ここで、引数 method は反実仮想生成の方法を表しています。
| 手法名 | メリット | デメリット |
|---|---|---|
| `genetic` | 連続値・複雑な条件でも柔軟に探索しやすい。 | 実行時間が長くなりやすい。 |
| `random` | 実装や挙動が直感的でわかりやすい。 | 良い反実仮想を安定して見つけにくい。|
| `kdtree` | 学習データに近い候補を探すため、実在しそうな反実仮想を得やすい。 | 学習データ内に適切な近傍がないと候補が見つかりにくい。 |

今回はデータ数もそこまで大きいわけではないのでgeneticを使っています。

### 介入特徴量の設定
DiCEは探索時に変動させる特徴量を事前に選ぶことができるため、非現実的な反実仮想シナリオが生成されないように制御することが可能です。本記事では以降介入特徴量と呼称します。

各介入特徴量には値の変動に制約をかけることもでき、実務上の制約を満たさない値や定義上あり得ない値に変動しないようにすることが可能です。

今回は複数の特徴量に「値が増加方向に動かないようにする制約」と「値が０を下回らない制約」をかけました。

```python
actionable_features = [
    "net_fraction_of_revolving_burden",
    "net_fraction_of_installment_burden",
    "percentage_trades_with_balance",
    "nr_inquiries_in_last_6_months_log1p",
    "high_ratio_bank_share_log1p",
]

monotone_decrease_features = [
    "percentage_trades_with_balance",
    "nr_inquiries_in_last_6_months_log1p",
    "high_ratio_bank_share_log1p",
]
monotone_increase_features = []

domain_bounds = {
    "net_fraction_of_revolving_burden": (0.0, None),
    "net_fraction_of_installment_burden": (0.0, None),
    "percentage_trades_with_balance": (0.0, 100.0),
    "nr_inquiries_in_last_6_months_log1p": (0.0, None),
    "high_ratio_bank_share_log1p": (0.0, None),
}

```

### 実験設定
今回は以下の３つのケースで比較実験を行います。

（画像）

| 予測モデルの出力（リスク確率） | 位置づけ |
|---:|---|
| 0.870 | 明確に高リスク側にあるケース |
| 0.622 | 中程度に高リスクなケース |
| 0.599 | 高リスク判定ではあるが、しきい値に比較的近いケース |

### 結果
#### 明確に高リスクのケース






