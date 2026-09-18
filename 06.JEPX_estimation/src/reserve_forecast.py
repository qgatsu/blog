"""一次調整力の約定価格を予測する。

入札は実需給前日 11:30-14:00 なので、その時点で使える情報だけを使う:
  - 調整力必要量 (募集量): 事前公表されるため既知
  - 過去の約定価格: D-1 は前日 15 時に約定済みで既知
  - 気象予報 / カレンダー
  - 同日のスポット価格: 前日 10:30 に約定確定済みなので既知
    (調整力の入札は 14 時なので、スポットの結果を見てから出せる)

現行制度 (前日取引・30分単位) のデータは 2026-03 以降のみで 153 日しかない。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_features() -> pd.DataFrame:
    r = pd.read_parquet(DATA_DIR / "reserve_primary_tokyo.parquet").sort_index()
    r = r[r.reserve_price.notna()].copy()

    df = pd.DataFrame(index=r.index)
    df["target"] = r["reserve_price"]

    # 事前公表される募集量 (入札時点で既知)
    df["required_mw"] = r["required_mw"]
    df["required_lag1d"] = r["required_mw"].shift(48)

    # 価格ラグ: D-1 は前日 15 時に約定済み
    for lag in (1, 2, 7):
        df[f"price_lag{lag}d"] = r["reserve_price"].shift(48 * lag)
    dates = r.index.normalize()
    daily = r["reserve_price"].groupby(dates).agg(["mean", "max", "min", "std"])
    for sh, pre in ((1, "prev_day"), (7, "prev_week")):
        s = daily.shift(sh)
        s.columns = [f"{pre}_{c}" for c in s.columns]
        df = df.join(s.reindex(dates).set_axis(r.index))
    # 落札量/応札量は事後情報なのでラグのみ
    df["cleared_lag1d"] = r["cleared_mw"].shift(48)
    df["offered_lag1d"] = r["offered_mw"].shift(48)
    df["tightness_lag1d"] = (r["offered_mw"] / r["required_mw"].replace(0, np.nan)).shift(48)

    # カレンダー
    df["hour"] = df.index.hour
    df["slot"] = df.index.hour * 2 + (df.index.minute // 30)
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month

    # 気象予報とスポット価格予測 (どちらも入札時点で既知)
    feat = pd.read_parquet(DATA_DIR / "features_gateclose.parquet")
    for c in ("temperature_2m", "shortwave_radiation", "net_demand_lag2d"):
        if c in feat.columns:
            df[c] = feat[c].reindex(df.index)
    pred = pd.read_parquet(DATA_DIR / "predictions_gateclose.parquet")
    df["spot_pred"] = pred["pred_q50"].reindex(df.index)
    spot_daily = pred["pred_q50"].groupby(pred.index.normalize()).mean()
    df["spot_pred_daymean"] = df.index.normalize().map(spot_daily)

    return df.dropna(subset=["target"])


def main():
    df = build_features()
    feats = [c for c in df.columns if c != "target"]
    days = df.index.normalize().unique()
    n = len(days)
    test_days = days[-31:]
    val_days = days[-61:-31]
    tr = ~df.index.normalize().isin(list(test_days) + list(val_days))
    va = df.index.normalize().isin(val_days)
    te = df.index.normalize().isin(test_days)
    print(f"全 {n} 日 / train {tr.sum()//48} 日 · val {va.sum()//48} 日 · test {te.sum()//48} 日")
    print(f"test 期間 {df.index[te].min().date()} ~ {df.index[te].max().date()}\n")

    y_te = df.loc[te, "target"].to_numpy()

    def report(name, pred):
        p = np.asarray(pred, float)
        m = ~np.isnan(p)
        mae = np.mean(np.abs(y_te[m] - p[m]))
        rmse = np.sqrt(np.mean((y_te[m] - p[m]) ** 2))
        return dict(model=name, mae=mae, rmse=rmse, mae_ratio=mae / y_te[m].mean() * 100)

    rows = [
        report("persistence D-1", df.loc[te, "price_lag1d"]),
        report("persistence D-7", df.loc[te, "price_lag7d"]),
        report("前日の日平均", df.loc[te, "prev_day_mean"]),
        report("全期間平均(定数)", np.full(te.sum(), df.loc[tr, "target"].mean())),
    ]

    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbosity=-1, seed=42, num_threads=0)
    dtr = lgb.Dataset(df.loc[tr, feats], label=df.loc[tr, "target"])
    dva = lgb.Dataset(df.loc[va, feats], label=df.loc[va, "target"], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    pred_lgb = model.predict(df.loc[te, feats])
    rows.append(report(f"LightGBM (iter={model.best_iteration})", pred_lgb))

    # 募集量を落としたモデル (事前公表情報の寄与を見る)
    feats_norq = [c for c in feats if not c.startswith("required")]
    m2 = lgb.train(params, lgb.Dataset(df.loc[tr, feats_norq], label=df.loc[tr, "target"]),
                   num_boost_round=2000,
                   valid_sets=[lgb.Dataset(df.loc[va, feats_norq], label=df.loc[va, "target"])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    rows.append(report("LightGBM (募集量なし)", m2.predict(df.loc[te, feats_norq])))

    res = pd.DataFrame(rows)
    print(f"{'モデル':26s}{'MAE':>8s}{'RMSE':>8s}{'MAE/平均':>10s}")
    for r in rows:
        print(f"{r['model']:26s}{r['mae']:8.3f}{r['rmse']:8.3f}{r['mae_ratio']:9.1f}%")

    base = [r for r in rows if r["model"] == "persistence D-1"][0]["mae"]
    best = min(rows, key=lambda r: r["mae"])
    print(f"\n最良: {best['model']}  MAE {best['mae']:.3f} "
          f"(persistence D-1 比 {(best['mae']/base-1)*100:+.1f}%)")

    print("\n=== 特徴量重要度 (gain, 上位10) ===")
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    for k, v in imp.head(10).items():
        print(f"  {k:24s} {v:10,.0f}")

    out = pd.DataFrame({"target": y_te, "pred_lgbm": pred_lgb,
                        "pred_persistence": df.loc[te, "price_lag1d"].to_numpy()},
                       index=df.index[te])
    out.to_parquet(DATA_DIR / "reserve_predictions.parquet")
    res.to_csv(DATA_DIR / "reserve_forecast_metrics.csv", index=False)
    print("\nsaved reserve_predictions.parquet")


if __name__ == "__main__":
    main()
