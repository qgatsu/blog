"""ベースラインの確定と評価。

予測モデル: RMSE (L2) 目的の LightGBM。前日10時の情報境界を守った gateclose 特徴量を使う。
評価は 2 系統を並べる:
  予測精度   MAE / rMAE / RMSE / sMAPE        … EPF の標準指標 (Lago et al. 2021)
  意思決定   VCR / Kendall tau / Spearman     … 蓄電池最適化の指標 (arXiv 2604.12082)

VCR (Value Capture Ratio) = その予測での収益 / 真の価格で計画したときの収益。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_backtest import solve_day, settle

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def smape(y, p, eps=1e-3):
    return float(np.mean(2 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), eps)) * 100)


def eval_forecast(y, p, naive_mae):
    e = y - p
    return dict(mae=float(np.mean(np.abs(e))), rmae=float(np.mean(np.abs(e)) / naive_mae),
                rmse=float(np.sqrt(np.mean(e ** 2))), smape=smape(y, p))


def eval_decision(df: pd.DataFrame, col: str, oracle_profit: float) -> dict:
    profit, taus, rhos = 0.0, [], []
    for _, g in df.groupby(df.index.normalize()):
        if len(g) != 48:
            continue
        true = g["target"].to_numpy()
        pred = np.maximum(g[col].to_numpy(), 0.01)
        plan = solve_day(pred)
        if plan is None:
            continue
        profit += settle(plan[0], plan[1], true)["gross"]
        taus.append(kendalltau(true, pred).statistic)
        rhos.append(spearmanr(true, pred).statistic)
    n = df.index.normalize().nunique()
    return dict(profit_per_day=profit / n, vcr=profit / oracle_profit,
                kendall=float(np.nanmean(taus)), spearman=float(np.nanmean(rhos)))


def main() -> None:
    d = pd.read_parquet(DATA_DIR / "predictions_gateclose.parquet")
    f = pd.read_parquet(DATA_DIR / "features_gateclose.parquet")
    d["persistence_1d"] = f.loc[d.index, "lag_1d_same_slot"]
    d["persistence_7d"] = f.loc[d.index, "lag_7d_same_slot"]
    d["oracle"] = d["target"]

    y = d["target"].to_numpy()
    naive_mae = float(np.mean(np.abs(y - d["persistence_7d"].to_numpy())))

    # オラクル収益 (VCR の分母)
    orc = eval_decision(d, "oracle", 1.0)
    oracle_profit = orc["profit_per_day"] * d.index.normalize().nunique()
    print(f"期間 {d.index.min().date()} ~ {d.index.max().date()}  "
          f"{d.index.normalize().nunique()} 日\n")

    models = {
        "オラクル(真の価格)": "oracle",
        "LightGBM (L2) ★基準": "pred_point",
        "LightGBM (q50)": "pred_q50",
        "persistence D-1": "persistence_1d",
        "persistence D-7": "persistence_7d",
    }

    rows = []
    for label, col in models.items():
        fc = eval_forecast(y, d[col].to_numpy(), naive_mae)
        dc = eval_decision(d, col, oracle_profit)
        rows.append(dict(model=label, **fc, **dc))
    res = pd.DataFrame(rows)

    print("=" * 96)
    print("予測精度（EPF の標準指標）と 意思決定（蓄電池）の比較")
    print("=" * 96)
    print(f"{'モデル':22s}{'MAE':>7s}{'rMAE':>7s}{'RMSE':>7s}{'sMAPE':>8s}"
          f"{'  |':>3s}{'Kendall':>9s}{'Spearman':>10s}{'収益/日':>10s}{'VCR':>8s}")
    for r in rows:
        print(f"{r['model']:22s}{r['mae']:7.2f}{r['rmae']:7.3f}{r['rmse']:7.2f}{r['smape']:8.1f}"
              f"{'  |':>3s}{r['kendall']:9.3f}{r['spearman']:10.3f}"
              f"{r['profit_per_day']:10,.0f}{r['vcr']*100:7.1f}%")

    # 指標間の順位一致を見る
    print("\n各指標で順位づけしたときの並び (良い順):")
    for m, asc in (("mae", True), ("rmse", True), ("kendall", False), ("vcr", False)):
        order = res.sort_values(m, ascending=asc)["model"].tolist()
        print(f"  {m:9s}: " + " > ".join(x.replace(" ★基準", "") for x in order))

    res.to_csv(DATA_DIR / "baseline_metrics.csv", index=False)
    (DATA_DIR / "baseline_meta.json").write_text(json.dumps(
        {"period": [str(d.index.min()), str(d.index.max())],
         "days": int(d.index.normalize().nunique()),
         "oracle_profit_per_day": oracle_profit / d.index.normalize().nunique(),
         "naive_mae_for_rmae": naive_mae}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved baseline_metrics.csv")


if __name__ == "__main__":
    main()
