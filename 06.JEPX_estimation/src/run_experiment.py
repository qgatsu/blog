"""actual (事後データ) と gateclose (前日10時締切制約) を同一条件で比較する。

点予測と分位点予測 (q10/q50/q90) を両データセットで学習し、
精度指標 (RMSE/MAE/sMAPE/Pinball) とカバレッジを出す。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_MONTHS, VAL_MONTHS = 6, 3
QUANTILES = [0.1, 0.5, 0.9]
SEED = 42

LGBM_PARAMS = dict(learning_rate=0.02, num_leaves=63, max_depth=-1,
                   min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, num_threads=0, verbosity=-1, seed=SEED)
NUM_ROUNDS, EARLY_STOP = 3000, 100

EXCLUDE = {"target", "target_log1p", "target_arsinh", "target_diff_1d", "target_diff_7d",
           "target_dev_prev_day", "target_dev_prev_week"}


def smape(y, p, eps=1e-3):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), eps)) * 100)


def pinball(y, p, q):
    d = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(y, p, model, split, dataset):
    e = np.asarray(y, float) - np.asarray(p, float)
    return dict(dataset=dataset, model=model, split=split, n=len(y),
                rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                smape=smape(y, p))


def split_frame(df: pd.DataFrame, last_ts: pd.Timestamp):
    test_start = (last_ts - pd.DateOffset(months=TEST_MONTHS)).normalize()
    val_start = (test_start - pd.DateOffset(months=VAL_MONTHS)).normalize()
    return (df[df.index < val_start], df[(df.index >= val_start) & (df.index < test_start)],
            df[df.index >= test_start], val_start, test_start)


def train_lgbm(Xtr, ytr, Xva, yva, feats, objective=None, alpha=None):
    params = {**LGBM_PARAMS}
    if objective == "quantile":
        params.update(objective="quantile", alpha=alpha, metric="quantile")
    else:
        params.update(objective="regression", metric="rmse")
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=feats)
    dva = lgb.Dataset(Xva, label=yva, feature_name=feats, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def main():
    meta = json.loads((DATA_DIR / "dataset_meta.json").read_text())
    # 両データセットを同一期間に揃える (予報が使える 2021-04-01 以降)
    common_start = pd.Timestamp(meta["gateclose"]["period"][0])

    rows, preds_out, models_info = [], {}, {}
    for name in ("era5_actual", "analysis", "fc_weather", "gateclose"):
        df = pd.read_parquet(DATA_DIR / f"features_{name}.parquet")
        df = df[df.index >= common_start]
        feats = [c for c in df.columns if c not in EXCLUDE]
        tr, va, te, val_start, test_start = split_frame(df, df.index.max())
        print(f"\n=== {name} ===  features={len(feats)}")
        print(f"  train {tr.index.min().date()}~{tr.index.max().date()} n={len(tr):,} | "
              f"val ~{test_start.date()} n={len(va):,} | test ~{te.index.max().date()} n={len(te):,}")

        Xtr, ytr = tr[feats], tr["target"]
        Xva, yva = va[feats], va["target"]
        Xte, yte = te[feats], te["target"]

        # ベースライン: 前週同スロット
        for split, d in (("val", va), ("test", te)):
            rows.append(evaluate(d["target"], d["lag_7d_same_slot"], "seasonal_naive_d7", split, name))

        # 点予測
        m = train_lgbm(Xtr, ytr, Xva, yva, feats)
        pv, pt = m.predict(Xva), m.predict(Xte)
        rows.append(evaluate(yva, pv, "lgbm_point", "val", name))
        rows.append(evaluate(yte, pt, "lgbm_point", "test", name))
        models_info[f"{name}_point_iter"] = m.best_iteration
        pred_te = {"target": yte.to_numpy(), "pred_point": pt}

        # 分位点
        for q in QUANTILES:
            mq = train_lgbm(Xtr, ytr, Xva, yva, feats, objective="quantile", alpha=q)
            pqv, pqt = mq.predict(Xva), mq.predict(Xte)
            pred_te[f"pred_q{int(q*100):02d}"] = pqt
            for split, yy, pp in (("val", yva, pqv), ("test", yte, pqt)):
                r = evaluate(yy, pp, f"lgbm_q{int(q*100):02d}", split, name)
                r["pinball"] = pinball(yy, pp, q)
                rows.append(r)

        # 80% 区間のカバレッジ
        lo, hi = pred_te["pred_q10"], pred_te["pred_q90"]
        cov = float(np.mean((yte.to_numpy() >= lo) & (yte.to_numpy() <= hi)) * 100)
        width = float(np.mean(hi - lo))
        models_info[f"{name}_coverage80_test"] = cov
        models_info[f"{name}_interval_width_test"] = width
        print(f"  80%区間 カバレッジ {cov:.1f}%  平均幅 {width:.2f} 円/kWh")

        preds_out[name] = pd.DataFrame(pred_te, index=te.index)
        preds_out[name].to_parquet(DATA_DIR / f"predictions_{name}.parquet")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "metrics_comparison.csv", index=False)
    (DATA_DIR / "models_info.json").write_text(json.dumps(models_info, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("test 期間の比較 (同一期間・同一分割)")
    print("=" * 78)
    piv = res[res.split == "test"].pivot_table(index="model", columns="dataset",
                                               values=["rmse", "mae", "smape"])
    print(piv.round(3).to_string())
    print("\n" + "=" * 78)
    print("条件間の差分 (test RMSE の変化率)")
    print("=" * 78)
    steps = [("era5_actual", "analysis", "気象データソースの差 (ERA5 -> 予報モデル)"),
             ("analysis", "fc_weather", "気象予報誤差の影響"),
             ("fc_weather", "gateclose", "需給情報が D-2 までに限られる影響"),
             ("analysis", "gateclose", "締切制約の合計")]
    t = res[res.split == "test"].pivot_table(index="model", columns="dataset", values="rmse")
    for a, b, label in steps:
        if {a, b} <= set(t.columns):
            deg = (t[b] / t[a] - 1) * 100
            print(f"  {label}")
            print(f"    " + "  ".join(f"{i}={v:+.1f}%" for i, v in deg.items()))


if __name__ == "__main__":
    main()
