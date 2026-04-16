---
title: groupShapleyのすすめ
tags:
  - 機械学習
  - SHAP
  - XAI
private: false
updated_at: '2026-04-14T17:26:10+09:00'
id: 10ef2a77ca7ef3f69a57
organization_url_name: null
slide: false
ignorePublish: false
---

# 注意書き

この記事は,著者が個人開発の中で調べたことや試したことを整理した備忘録として書いています.  
あわせて技術記事という性格上,実際に手元で試す場合は原典や実装,前提条件を確認したうえで各自の責任で扱ってください.

# はじめに

機械学習モデルの解釈手法としてSHAP はかなりメジャーな選択肢です.特徴量ごとの寄与をサンプル単位でも全体傾向でも見られるためとりあえず SHAP を見ておく,という場面はかなり多いように思えます.

ただ,テーブルデータではカテゴリ変数をそのままモデルに入れずencoding によって複数列へ分解してから学習させることがあります.one-hot encoding[^onehot] はその代表例で,実装上は扱いやすい一方で解釈の段階では元の変数との対応が少し見えづらくなります.

カテゴリ変数を one-hot encoding してモデルに入れている場合,通常の SHAP で得られるのは **「居住地域が東京である」** のような one-hot 化された列ごとの寄与です.たとえば **「居住地域」や「契約プラン」** のようなカテゴリ項目を見たいのに,実際には各候補値ごとの列に分かれてしまうため, **元のカテゴリ変数そのものがどれくらい効いたのか** は直感的に見えにくくなります.

さらにこの結果を非エンジニアの第三者に説明しようとすると,「まず one-hot encoding という前処理があって...」という話から始める必要があり,分析そのものとは別の説明コストが発生します.

そこで本記事では,one-hot 後の列を元の特徴グループ単位で扱う groupShapley に注目します.まず通常 SHAP の考え方を整理し,つぎに groupShapley が何をしているのかを見たうえで,年収予測データを使ってグループ単位の解釈を試します.

# そもそもSHAPとは

SHAP は,ある予測に対して各特徴量がどれだけ寄与したかを協力ゲーム理論の Shapley value をベースに割り当てる考え方です.[^shap_paper] ざっくり言えば, **「特徴量がチームで予測を作っているとして,その特徴が平均的にどれだけ貢献したか」を公平に分配する** 手法と見るとわかりやすいです.

入力を $ x = (x_1, \dots, x_M) $ ,モデルを $ f $ とすると,特徴量 $ i $ の SHAP 値 $ \phi_i $ は概念的には次の形で定義されます.

```math
\phi_i =
\sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!(M-|S|-1)!}{M!}
\left[
v(S \cup \{i\}) - v(S)
\right]
```

ここで $ N = \{1,\dots,M\} $ は全特徴量の集合で, $ v(S) $ は「特徴量集合 $ S $ だけがわかっているときのモデル出力の期待値」と考えれば十分です.つまり SHAP はある特徴量を coalition[^coalition] に加えたときに予測がどれだけ変わるかを,全ての参加順序で平均していることになります.

この定義からSHAP には次のようなうれしい性質があります.

- 寄与の総和が予測値とベースラインの差に一致する.
- 同じ働きをする特徴量には同じ寄与が割り当てられる.
- まったく影響しない特徴量の寄与は 0 になる.

一方で,実際に全ての部分集合を厳密に評価するのは重いため実装ではモデルに応じて TreeSHAP や Sampling 系の近似が使われます.今回の通常 SHAP では XGBoost に対して `TreeExplainer`[^treeexplainer] を使うので,木モデルに適した高速な計算ができます.

# groupShapleyとは

groupShapley は特徴量を1列ずつではなく **あらかじめ定義したグループ単位でまとめて寄与を測る** 考え方です.[^groupshapley_paper] 今回のようにカテゴリ変数を one-hot 化している場合は,たとえば **あるカテゴリ項目から展開された複数列を,元の1変数としてひとかたまりで扱う** イメージです.

通常 SHAP が各 one-hot 列 $ x_j $ に対して $ \phi_j $ を割り当てるのに対し,groupShapley ではグループ $ G_k $ に対して $ \phi_{G_k} $ を考えます.直感的には「あるカテゴリ項目全体が予測にどう効いたか」を見たいので,coalition の単位を one-hot 列ではなく元特徴のグループに持ち上げるイメージです.

本記事では `PartitionExplainer` を使って coalition をグループ単位に制約し,groupShapley を近似しますが,この計算は通常の SHAP よりかなり重くなります.理由は大きく2つあります.

- `TreeExplainer` のような木モデル特化の高速計算ではなく,マスキングを伴う近似計算になる.
- 各サンプルについて,background[^background] データを参照しながら複数回モデル評価を行う必要がある.

パラメータを調整すると基準分布や近似精度は安定しやすくなりますがそのぶん計算量は増えます.そのため **どこまで厳密に見るか** と **どこまで計算時間を許容するか** のバランスが重要になります.

# 分析

ここからはAdult / Census Income データで実際に groupShapley を見ていきます.

## データ

今回は UCI Adult (Census Income) データを使います.目的変数は年収が `$ >50K $` かどうかの2値分類です.学習データ `adult.data` とテストデータ `adult.test` を zip から直接読み込み,カテゴリ列は欠損を `Unknown` で補完したうえで one-hot encoding します.

特徴量の概要は以下の通りです.

| 項目 | 内容 |
| --- | --- |
| 数値列 | `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| カテゴリ列 | `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |

カテゴリ列の意味を簡単にまとめると次の通りです.

| 列名 | 意味 | 主な値の例 |
| --- | --- | --- |
| `workclass` | 就業形態 | `Private`, `Self-emp-not-inc`, `Local-gov` |
| `education` | 最終学歴 | `Bachelors`, `HS-grad`, `Masters` |
| `marital_status` | 婚姻状況 | `Never-married`, `Married-civ-spouse`, `Divorced` |
| `occupation` | 職種 | `Prof-specialty`, `Exec-managerial`, `Craft-repair` |
| `relationship` | 世帯内の立場 | `Husband`, `Not-in-family`, `Own-child` |
| `race` | 人種カテゴリ | `White`, `Black`, `Asian-Pac-Islander` |
| `sex` | 性別 | `Male`, `Female` |
| `native_country` | 出身国 | `United-States`, `Mexico`, `Philippines` |

ここで重要なのは,たとえば `occupation` は元々1列ですが,学習時には複数の one-hot 列に分解されることです.今回の groupShapley では,この分解後の列群を再び `occupation` という1つのグループとして扱います.

## 予測モデルと精度

分類モデルには XGBoost を使います.今回は以下のような設定でベースラインモデルを学習します.

```python
baseline_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)
```

ここでは精度指標の参考として Accuracy と ROC AUC[^roc_auc] を記載します.

| Metric | Value |
| --- | ---: |
| Accuracy | `0.8758` |
| ROC AUC | `0.9278` |


## groupShapleyの実装

SHAP には公式ドキュメントや API リファレンスがあり,今回使う `PartitionExplainer` や `maskers.Partition` もその流れで利用できます.

- SHAP documentation: [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- API reference: [shap.PartitionExplainer](https://shap.readthedocs.io/en/stable/generated/shap.PartitionExplainer.html)
- API reference: [shap.maskers.Partition](https://shap.readthedocs.io/en/stable/generated/shap.maskers.Partition.html)

通常の TreeSHAP と違い,ここでは **説明時の coalition 構造をこちらで設計** します.

### まず通常 SHAP を確認する

比較対象として先に通常の SHAP を計算します.こちらは one-hot 後の列ごとの寄与です.

```python
X_shap = X_test.copy()
explainer = shap.TreeExplainer(baseline_model)
shap_values = explainer(X_shap)
```

summary plot では,`capital_gain` のような数値列に加えて`marital_status_*` や `relationship_*` のような one-hot 列が上位に並びます.

![normalshap_nocolor.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/30ba5c2a-42b3-4b29-a818-446282cced54.png)

bar plot にすると各列の平均絶対 SHAP 値は見やすくなりますが,カテゴリ変数が細かい列に分かれているためたとえば「結局 `occupation` 全体として重要なのか」は即答しづらいです.

### PartitionExplainer で groupShapley を近似する

ここで本題の groupShapley に入ります.まず one-hot 列からグループ構造を作り直し,`PartitionExplainer`[^partitionexplainer] に渡す linkage を自前で組みます.

```python
GROUP_SHAP_BACKGROUND_SIZE = 200
GROUP_SHAP_EXPLAIN_SIZE = 1000
GROUP_SHAP_MAX_EVALS = 500
GROUP_SHAP_BATCH_SIZE = 50

group_feature_groups = {col: [col] for col in numeric_cols}
for col in categorical_cols:
    group_feature_groups[col] = [
        encoded_col
        for encoded_col in X_train.columns
        if encoded_col.startswith(f"{col}_") and encoded_col not in numeric_cols
    ]
```

ここでの設定は次のような意味を持ちます.

- `BACKGROUND_SIZE`: マスキング時の参照分布として使うサンプル数
- `EXPLAIN_SIZE`: 実際に groupShapley を計算して集計する評価対象サンプル数
- `MAX_EVALS`: 1サンプルあたりの評価回数上限

次に,特徴グループ単位で結合する linkage を組み,group 用の masker と explainer を作ります.

```python
group_masker = shap.maskers.Partition(background_group, clustering=group_linkage)
group_explainer = shap.Explainer(
    predict_proba_positive,
    masker=group_masker,
    algorithm="partition",
    feature_names=group_feature_names,
)
```

そして得られた各列の寄与を同じグループに属する列ごとに合計して group 単位の寄与 `phi` を作ります.

```python
phi = pd.DataFrame(
    {
        group_name: shap_detail.values[:, [group_feature_index[col] for col in cols]].sum(axis=1)
        for group_name, cols in group_feature_groups.items()
    },
    index=explain_df.index,
)
importance = phi.abs().mean().sort_values(ascending=False)
```

この流れにより`occupation` や `education` といった元の変数単位で **mean(|group Shapley value|)** を比較できます.

![groupshap1000.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/ca0372db-901f-4fae-92f3-440b668e5949.png)

また,通常 SHAP 側の bar plot も同じグループ色で塗り分けると,groupShapley で重要だった変数の内側で,どの one-hot 列が効いているかを追いやすくなります.

![normalshap.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/96456fd6-fd53-45ea-8fa5-330930832e49.png)

この2つを並べて見ると,まず groupShapley で大きな変数群をつかみ,つぎに通常 SHAP でその内訳を見る,という導線がかなり自然になります.

## データサイズによる精度の分析

groupShapley は計算が重いため,全件に対して常に厳密に計算するのはつらい場面があります.そこでここでは,background や linkage を固定したまま,説明対象サンプル数 `explain_size` だけを変えて groupShapley の集計結果を比較します.

対象は以下の4パターンです.

```python
EXPLAIN_SIZE_GRID = [100, 500, 1000, 5000]
```

実行時の進捗出力をもとに,実行時間を整理すると次のようになります.

| `explain_size` | 実測時間[秒] | おおよその速度 |
| --- | ---: | ---: |
| `100` | `25` | 約2.45 it/s |
| `500` | `137` | 約3.40 it/s |
| `1000` | `274` | 約3.54 it/s |
| `5000` | `1259` | 約3.94 it/s |

`1000` 件で `274` 秒,`5000` 件では `1259` 秒なので,このセクションだけでも groupShapley が通常 SHAP よりかなり重いことがわかります.一方で,速度そのものは大きく悪化していないため,計算時間の増加はほぼ説明対象件数に比例していると見てよさそうです.

以下はサンプルサイズを変えていった場合のSHAP値の変動です.

![groupshap_gif.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/4383341/9c2d270c-3786-4405-b152-4bfab49248c5.gif)

比較的安定して見える理由としては,次の点が効いていると考えられます.

- 今回比較しているのは各サンプルの値そのものではなく`mean(|group Shapley value|)`[^mean_abs_group] という集計量なので,個々の揺れが平均化されやすい.
- 絶対値を取ってから平均しているため,寄与の向きよりも寄与の強さが前面に出やすく順位が崩れにくい.
- Adult データでは上位特徴のシグナルが比較的強くサンプル数を減らしても大局的な傾向が残りやすい.
- 今回の比較では background や linkage を固定しており,変えているのが explanation 対象件数だけなので,変動要因が限定されている.

もちろん希少カテゴリの影響を見たい場合や上位特徴どうしの差がごく小さい場合には,同じ設定でも順位がもう少し不安定になる可能性があります.

## 考察

ここまでの結果から,今回の分析で主に言いたいことは次の2点です.

1つ目は, **groupShapley の集計結果はデータや設定に依存するものの,今回のような条件では大局的な傾向は比較的安定して見えやすい** ということです.とくに「どの変数群が上位に来るか」というレベルでは,少数サンプルでも大まかな構図を掴める場合があります.

2つ目は,それでも groupShapley は通常 SHAP より明らかに重いため,実分析では「いつも最大件数で回す」のが最適とは限らないことです.探索段階では `explain_size=500` や `1000` で全体傾向を掴み,最終確認だけ大きめのサンプルで回す,といった運用のほうが現実的でしょう.

また,groupShapley だけで完結させるより, **groupShapley で変数群を掴み,通常 SHAP で群の内部を見る** という2段構えのほうが説明しやすいです.非エンジニアへの共有でも,最初に `occupation` や `marital_status` といった元の変数名で話を始められるため,one-hot の実装都合をいきなり説明しなくて済みます.

# まとめ

本記事ではone-hot encoding されたカテゴリ変数を通常 SHAP だけで読むときの見づらさを出発点に,groupShapley を使って元の特徴グループ単位で寄与を解釈する流れを整理しました.

通常 SHAP は高速で細かい内訳が見やすい一方でカテゴリ変数が分解されると説明コストが上がります.それに対して groupShapley は計算は重いものの,**まず元変数単位で重要な塊を掴める** という利点があります.実務では,groupShapley で大きな構図を見てから通常 SHAP に降りていく流れが使いやすそうです.

[^shap_paper]: SHAP の基本的な考え方は, Lundberg, Scott M., and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems 30*, 2017. に基づきます.
[^groupshapley_paper]: groupShapley の基本的な定式化は, Jullum, Martin, Annabelle Redelmeier, and Kjersti Aas. "groupShapley: Efficient prediction explanation with Shapley values for feature groups." *arXiv preprint arXiv:2106.12228*, 2021. を参照しています.
[^onehot]: one-hot encoding は,1つのカテゴリ変数を値ごとの 0/1 列に展開する前処理です.たとえば「契約プラン = A/B/C」を `plan_A`, `plan_B`, `plan_C` のような列に分けて表現します.
[^coalition]: coalition は,SHAP の文脈では「いま説明に参加している特徴量の集合」を指します.
[^treeexplainer]: `TreeExplainer` は,決定木や勾配ブースティング木のような木ベースモデルに対して SHAP 値を効率よく計算するための実装です.
[^background]: background は,特徴を隠したときに何を基準に補うかを決める参照データです.複数サンプルを使って期待値を取ることで説明の基準分布を表現します.
[^partitionexplainer]: `PartitionExplainer` は,特徴量どうしのまとまりを指定しながら SHAP 値を近似する explainer です.今回のようにグループ単位で寄与を見たいときに使いやすい実装です.
[^roc_auc]: ROC AUC は,しきい値を固定せずに分類モデルの順位付け性能をみる指標です.1 に近いほど正例を高く,負例を低くスコア付けできていると解釈できます.
[^mean_abs_group]: `mean(|group Shapley value|)` は,各サンプルについて計算した groupShapley 値の絶対値を取りその平均を取ったものです.寄与の向きではなく,どれだけ強く効いているかを集約して見ています.

# 参考文献

- Lundberg, Scott M., and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems 30*, 2017. https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Jullum, Martin, Annabelle Redelmeier, and Kjersti Aas. "groupShapley: Efficient prediction explanation with Shapley values for feature groups." *arXiv preprint arXiv:2106.12228*, 2021. https://arxiv.org/abs/2106.12228
