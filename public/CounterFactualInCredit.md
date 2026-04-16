---
title: DiCEによる反実仮想説明で与信審査AIを解釈してみよう
tags:
  - 機械学習
  - XAI
  - 反実仮想説明
private: false
updated_at: '2026-04-04T20:29:37+09:00'
id: 2c4fc0328d4033746ac6
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

Microsoft Research社開発の「DiCE」は反実仮想的なサンプルを生成することで、 **目的とした出力を得るための説明変数を直接出力** する手法です。

つまり「どういう要素をもとにその結果が出たのか」だけでなく **「どうすればその結果にならなかったのか」** を直接得ることができるため、その点でほかの手法より実際の意思決定に近い解釈アルゴリズムとなっています。

本記事では、HELOC[^heloc] の与信審査データを題材に、DiCEを使った反実仮想データ生成を通じて、施策決定につながる示唆が得られるかを見ていきます。

# DiCEについて
DiCEは一言で言えば **「AIの予測結果を望ましい方向に変えるためにどの特徴をどれだけ変えればよいかを、できるだけ少なく自然な変更で探す仕組み」** です。

![DiCE_gif.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/73f480a9-b3a2-4e58-93cb-280ea553b152.gif)
出典：https://interpret.ml/DiCE/

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
DiCEは内部で以下の最適化問題を解いています。
```math
C(x)=\arg \min _{c_1, \ldots, c_k} \frac{1}{k} \sum_{i=1}^k \operatorname{yloss}\left(f\left(c_i\right), y\right)+\frac{\lambda_1}{k} \sum_{i=1}^k \operatorname{dist}\left(c_i, x\right)-\lambda_2 \operatorname{dpp} \_\operatorname{diversity}\left(c_1, \ldots, c_k\right)
```

ここで$ \operatorname{yloss}\left(f\left(c_i\right), y\right) $ は **「望ましい出力と反実仮想サンプルを与えた場合の出力の差」** 、$ \operatorname{dist}\left(c_i, x\right) $ は **「対象の実サンプルとの距離」** 、$ \operatorname{dpp}\_\operatorname{diversity}\left(c_1, \ldots, c_k\right) $は **「各シナリオ同士がどれくらい多様性を持っているか」** を表しています。

そのため、上記最適化を解くことで **より少ない変更で結果を変動させる多様なシナリオ** を生成することができます。
 
# 問題設定
前置きが長くなりましたが、ここからは実データによる分析に入ります。
今回は与信審査AIプロジェクトにおいて以下のような課題を解決したいケースを想定します。
> A社では与信審査AIを運用しているが、いちどリスク有判定になったユーザーに再度の申し込みを促すため「どうすれば承認ラインに達するか」の目安を還元したい。

# データ
アメリカのアナリティクス企業FICOが公開しているHELOC(Home Equity Line of Credit) の与信審査データを使います。目的変数は一定期間内に深刻な延滞がある場合を$ 1 $、ない場合を$ 0 $としています。

また、施策に直結する特徴量については意味を後述しますが、全特徴量の定義や型などは以下のページを参照してください。

https://huggingface.co/datasets/mstz/heloc

# 使用モデル
今回は分類モデルとして、以下の記事で作成したロジスティック回帰モデルを用います。ざっくりとしたモデルの説明や参考リンクは以下の記事を参照してください。

https://qiita.com/na9atsuki/items/2b0f48f3157bc305b099

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

各介入特徴の意味を以下に示します。

| 特徴量 | 意味 |
|---|---|
| `net_fraction_of_revolving_burden` | リボルビング系与信[^revolving]の利用負担率 |
| `net_fraction_of_installment_burden` | 分割払い系与信の利用負担率 |
| `percentage_trades_with_balance` | 残高が残っている取引の比率 |
| `high_ratio_bank_share_log1p` | 高利用率状態にある銀行系取引の比率 |

各介入特徴量には値の変動に制約をかけることもでき、実務上の制約を満たさない値や定義上あり得ない値に変動しないようにすることが可能です。

今回は複数の特徴量に「値が増加方向に動かないようにする制約」と「値が０を下回らない制約」をかけました。

```python
actionable_features = [
    "net_fraction_of_revolving_burden",
    "net_fraction_of_installment_burden",
    "percentage_trades_with_balance",
    "high_ratio_bank_share_log1p",
]

monotone_decrease_features = [
    "percentage_trades_with_balance",
    "high_ratio_bank_share_log1p",
]
monotone_increase_features = []

domain_bounds = {
    "net_fraction_of_revolving_burden": (0.0, None),
    "net_fraction_of_installment_burden": (0.0, None),
    "percentage_trades_with_balance": (0.0, 100.0),
    "high_ratio_bank_share_log1p": (0.0, None),
}

def build_permitted_range(df: pd.DataFrame, query_df: pd.DataFrame, columns: list[str]) -> dict:
    permitted = {}
    query_row = query_df.iloc[0]

    for col in columns:
        lower = float(df[col].min())
        upper = float(df[col].max())

        if col in domain_bounds:
            domain_lower, domain_upper = domain_bounds[col]
            if domain_lower is not None:
                lower = max(lower, float(domain_lower))
            if domain_upper is not None:
                upper = min(upper, float(domain_upper))

        if col in monotone_decrease_features:
            upper = min(upper, float(query_row[col]))

        if col in monotone_increase_features:
            lower = max(lower, float(query_row[col]))

        permitted[col] = [lower, upper]

    return permitted
```

このように、単調制約は `permitted_range` を組み立てる段階で反映しています。

### 反実仮想生成の設定
反実仮想の生成時には、目標クラス、生成件数、到達判定のしきい値などを以下のように設定します。

```python
desired_risk_class = 0

case_generation_config = {
    "features": actionable_features,
    "total_CFs": 3,
    "diversity_weight": 20.0,
    "proximity_weight": 0.5,
    "stopping_threshold": best_threshold,
}

case_counterfactuals = dice.generate_counterfactuals(
    query_instances=case_query,
    total_CFs=case_generation_config["total_CFs"],
    desired_class=desired_risk_class,
    features_to_vary=case_generation_config["features"],
    permitted_range=case_permitted_range,
    stopping_threshold=case_generation_config["stopping_threshold"],
    diversity_weight=case_generation_config["diversity_weight"],
    proximity_weight=case_generation_config["proximity_weight"],
)
```

各引数の詳細な仕様や最新の対応状況は、公式ドキュメントとGitHubリポジトリも参照してください。

- Docs: https://interpret.ml/DiCE/
- GitHub: https://github.com/interpretml/DiCE

今回は `desired_class=0` として、リスク有判定のサンプルを承認側に反転させる設定にしています。また、`stopping_threshold=best_threshold`[^stopping_threshold] とすることで、到達判定もモデル本体のしきい値に揃えています。

### 実験設定
今回はモデルの best threshold である `0.37` を判定しきい値として用い、陽性判定サンプルの中から `risk_proba` が `0.8`、`0.6`、`0.4` に近い３ケースを選んで比較します。

![compare_sample.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/202c35cf-6cff-4599-ab25-4dada1b160f4.png)

| 選定基準 | 実際に選ばれた `risk_proba` | 位置づけ |
|---:|---:|---|
| 0.8 付近 | 0.870 | 明確に高リスク側にあるケース |
| 0.6 付近 | 0.599 | 中程度に高リスクなケース |
| 0.4 付近 | 0.431 | しきい値に比較的近いケース |

### 結果
#### 明確に高リスクのケース（risk_proba = 0.870）

|  | リボルビング系与信の利用負担率 | 分割払い系与信の利用負担率 | 残高が残っている取引の比率 | 高利用率状態にある銀行系取引の比率 |
|---|---:|---:|---:|---:|
| original | 99.0 | 38.0 | 100.0 | 0.1818 |
| シナリオ1 | 92.0 | 25.0 | 100.0 | 0.1052 |
| シナリオ2 | 82.0 | 42.0 | 100.0 | 0.1052 |
| シナリオ3 | 83.0 | 42.0 | 100.0 | 0.1052 |

- シナリオ1: 分割払い系与信の利用負担率を下げつつ、高利用率状態にある銀行系取引の比率も抑える案
- シナリオ2: リボルビング系与信の利用負担率を大きく下げつつ、高利用率状態にある銀行系取引の比率も抑える案
- シナリオ3: シナリオ2とほぼ同型で、主にリボルビング系与信の利用負担率の圧縮で判定反転を狙う案

#### 中程度に高リスクなケース（risk_proba = 0.599）

|  | リボルビング系与信の利用負担率 | 分割払い系与信の利用負担率 | 残高が残っている取引の比率 | 高利用率状態にある銀行系取引の比率 |
|---|---:|---:|---:|---:|
| original | 76.0 | 83.0 | 100.0 | 0.1000 |
| シナリオ1 | 76.0 | 74.0 | 100.0 | 0.0000 |
| シナリオ2 | 71.0 | 60.0 | 100.0 | 0.1052 |
| シナリオ3 | 30.0 | 58.0 | 100.0 | 0.1052 |

- シナリオ1: 分割払い系与信の利用負担率を少し下げつつ、高利用率状態にある銀行系取引の比率を抑える案
- シナリオ2: リボルビング系与信の利用負担率と分割払い系与信の利用負担率をともに下げる案
- シナリオ3: リボルビング系与信の利用負担率を大きく下げつつ、分割払い系与信の利用負担率も下げる案

#### しきい値に比較的近いケース（risk_proba = 0.431）

|  | リボルビング系与信の利用負担率 | 分割払い系与信の利用負担率 | 残高が残っている取引の比率 | 高利用率状態にある銀行系取引の比率 |
|---|---:|---:|---:|---:|
| original | 89.0 | 74.0 | 100.0 | 0.1429 |
| シナリオ1 | 76.0 | 74.0 | 100.0 | 0.1052 |
| シナリオ2 | 81.0 | 74.0 | 88.0 | 0.1052 |

- シナリオ1: リボルビング系与信の利用負担率を下げつつ、高利用率状態にある銀行系取引の比率も抑える案
- シナリオ2: リボルビング系与信の利用負担率を少し下げ、残高が残っている取引の比率も抑える案

#### 考察

今回の3ケースを見ると、DiCEは単に「負担率を下げるべき」といった抽象的な方向性だけでなく、**どの特徴をどの程度動かせば判定反転に届くのか** をケースごとに具体的なシナリオとして返してくれました。特に、しきい値に近いケースではリボルビング系与信の利用負担率や高利用率状態にある銀行系取引の比率を少し抑えるだけで済む案が見られた一方で、中程度に高リスクなケースでは複数特徴を同時に大きく動かす案が多く、ケースごとに必要な介入の重さが異なることが確認できます。

また、結果の中には **リボルビング系与信の利用負担率を下げる一方で、分割払い系与信の利用負担率がやや上がる** シナリオも含まれていました。これは、総負担を一律に下げる方向だけでなく、モデル上は「リボルビング側の圧縮を優先し、その一部が分割払い側に移る」ような構図でも判定反転候補になり得ることを示しています。もちろん、それが実際に顧客へ還元すべき助言として妥当かどうかは別途検討が必要ですが、少なくとも **モデルがどの種類の負債構成の変化を低リスク側と見ているか** を把握する材料にはなります。

一見すると、最も高リスクなケースのほうが中程度に高リスクなケースよりも少ない変化で判定反転しているようにも見えます。直感にはやや反しますが、これは **元のリスク確率の高さだけで難易度が決まるわけではなく、どの特徴に改善余地が残っているか** に強く依存しているためだと考えられます。今回の中程度に高リスクなケースでは、`percentage_trades_with_balance` がもともと `100.0` でほぼ動かず、`high_ratio_bank_share_log1p` も変化余地が限られていました。そのため、実質的にはリボルビング系与信と分割払い系与信の利用負担率に介入が集中し、より大きな変化が必要になったと解釈できます。反対に高リスクのケースでは、動かせる特徴の組み合わせにまだ余地があり、その分だけ少ない変更でも承認側に届くシナリオが見つかった可能性があります。

#### 課題

今回の実験では、介入特徴を4変数に絞り、3ケースだけを対象に見ているため、得られた傾向をそのまま一般化するのは難しいです。特に、どの特徴を「動かせる」とみなすかで反実仮想の形はかなり変わるため、実務に寄せるなら業務ルールに沿った制約設計が重要になります。

# まとめ

本記事では、与信審査モデルの解釈において、特徴量重要度だけでは施策に落とし込みにくいという課題に対して、DiCEによる反実仮想説明を試しました。回帰係数からはモデルがどの特徴をどう評価しているかは分かる一方で、実際に「何をどれだけ変えればよいか」は直接は分かりません。その点、DiCEを使うことで、介入可能な特徴だけを対象にしながら、判定反転に必要な変更案をシナリオとして具体化できることが確認できました。

もちろん、生成された反実仮想がそのまま実務上の助言として妥当とは限らず、顧客への還元には業務ルールや倫理面の検討も必要です。それでも、モデルの出力を施策検討につなげるための一段深い解釈として、反実仮想説明は有用な道具になりそうだと感じました。今後は、実務制約をより厳密に組み込んだり、別のモデルでも同様の傾向が出るかを比較したりしながら、もう少し掘ってみたいと思います。

[^heloc]: Home Equity Line of Credit の略で、住宅を担保に設定する与信枠のことです。
[^revolving]: ここでは主に、利用残高に応じて返済が続くリボルビング型の与信を指します。
[^stopping_threshold]: 反実仮想生成において、目標クラスに到達したとみなすための予測確率のしきい値です。

# 参考文献
[Explaining machine learning classifiers through diverse counterfactual explanations](https://dl.acm.org/doi/10.1145/3351095.3372850)
[Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://arxiv.org/abs/1905.07697)
