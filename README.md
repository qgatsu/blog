# Tech Blog Notebook

このディレクトリでは、テックブログのネタを notebook 形式で育てます。

## 基本方針

各テーマは以下の流れで整理します。

1. 具体的なビジネスクエスチョンを定義する
2. リサーチクエスチョンに還元する
3. 手法を紹介する
4. 必要なら論文実装する
5. 分析して示唆をまとめる

## ディレクトリ構成

```text
blog/
  README.md
  templates/
    article_notebook.ipynb
  topics/
    sample_business_question/
      README.md
      notebook.ipynb
      data/
      refs/
      src/
```

## 運用ルール

- 1テーマにつき `topics/<slug>/` を 1 つ作る
- notebook 本体は `notebook.ipynb`
- 補助コードは `src/`
- 参照論文やメモは `refs/`
- データは `data/` に置く

## 推奨執筆フロー

1. `templates/article_notebook.ipynb` をコピーして `topics/<slug>/notebook.ipynb` を作る
2. `topics/<slug>/README.md` にテーマの狙いと成果物を短く書く
3. notebook で仮説、実装、分析を進める
4. 内容が固まったら notebook をもとに記事化する

## Qiita CLI 起動コマンド
cd /home/kohei/WorkSpace/blog
npx qiita preview