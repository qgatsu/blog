"""ハイブリッド損失 (alpha, beta) と MLP のハイパーパラメータを Optuna で探索する。

選択基準は validation の VCR。予測精度ではなく意思決定の質で選ぶ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_rank import build_daily, train, evaluate, TEST_MONTHS, VAL_MONTHS

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
HIDDEN_CHOICES = {"256": (256,), "512-256": (512, 256),
                  "512-256-128": (512, 256, 128), "1024-512": (1024, 512)}


def main():
    df = pd.read_parquet(DATA_DIR / "features_gateclose.parquet")
    X, Y, di, sc, dc = build_daily(df)
    last = di.max()
    ts = (last - pd.DateOffset(months=TEST_MONTHS)).normalize()
    vs = (ts - pd.DateOffset(months=VAL_MONTHS)).normalize()
    tr, va, te = di < vs, (di >= vs) & (di < ts), di >= ts
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-8
    my, sy = Y[tr].mean(), Y[tr].std()
    Xs = ((X - mx) / sx).astype(np.float32)
    Ys = ((Y - my) / sy).astype(np.float32)

    orc_va = evaluate(Y[va], Y[va], 1.0)["profit_per_day"] * va.sum()
    orc_te = evaluate(Y[te], Y[te], 1.0)["profit_per_day"] * te.sum()
    print(f"train {tr.sum()} / val {va.sum()} / test {te.sum()} 日")
    print(f"選択基準: validation の VCR\n")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.05, 1.0)
        beta = trial.suggest_float("beta", 0.0, 1.0)
        hk = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES))
        dropout = trial.suggest_float("dropout", 0.0, 0.45)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        m = train(Xs[tr], Ys[tr], Xs[va], Ys[va], alpha, seed=0, beta=beta,
                  hidden=HIDDEN_CHOICES[hk], dropout=dropout, lr=lr, wd=wd)
        m.eval()
        with torch.no_grad():
            pv = m(torch.tensor(Xs[va])).numpy() * sy + my
        r = evaluate(pv, Y[va], orc_va)
        trial.set_user_attr("val_mae", r["mae"])
        trial.set_user_attr("val_kendall", r["kendall"])
        return r["vcr"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"探索 {len(study.trials)} 試行")
    print(f"best val VCR = {study.best_value*100:.2f}%")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k:10s} {v if isinstance(v,str) else round(v,5)}")

    # 上位 5 件
    print("\n上位5試行:")
    top = sorted(study.trials, key=lambda t: -(t.value or 0))[:5]
    print(f"{'VCR':>7s}{'alpha':>8s}{'beta':>7s}{'hidden':>13s}{'drop':>6s}"
          f"{'lr':>9s}{'MAE':>7s}{'Kendall':>9s}")
    for t in top:
        p = t.params
        print(f"{t.value*100:7.2f}{p['alpha']:8.3f}{p['beta']:7.3f}{p['hidden']:>13s}"
              f"{p['dropout']:6.2f}{p['lr']:9.5f}"
              f"{t.user_attrs.get('val_mae',0):7.2f}{t.user_attrs.get('val_kendall',0):9.3f}")

    # best を複数シードで test 評価
    bp = study.best_params
    print("\ntest 評価 (5 シード):")
    vals = []
    for s in range(5):
        m = train(Xs[tr], Ys[tr], Xs[va], Ys[va], bp["alpha"], seed=s, beta=bp["beta"],
                  hidden=HIDDEN_CHOICES[bp["hidden"]], dropout=bp["dropout"],
                  lr=bp["lr"], wd=bp["wd"])
        m.eval()
        with torch.no_grad():
            pt = m(torch.tensor(Xs[te])).numpy() * sy + my
        r = evaluate(pt, Y[te], orc_te)
        vals.append(r)
        print(f"  seed {s}: VCR {r['vcr']*100:5.2f}%  MAE {r['mae']:5.2f}  "
              f"Kendall {r['kendall']:.3f}  収益 {r['profit_per_day']:,.0f} 円/日")
    v = [r["vcr"] * 100 for r in vals]
    print(f"\n  平均 VCR {np.mean(v):.2f}% ± {np.std(v):.2f}  "
          f"MAE {np.mean([r['mae'] for r in vals]):.2f}  "
          f"Kendall {np.mean([r['kendall'] for r in vals]):.3f}  "
          f"収益 {np.mean([r['profit_per_day'] for r in vals]):,.0f} 円/日")
    print(f"\n  比較: LightGBM(L2) 64.7% / 手動 α=0.3 67.66%")

    study.trials_dataframe().to_csv(DATA_DIR / "search_trials.csv", index=False)
    (DATA_DIR / "search_best.json").write_text(json.dumps(
        {"best_params": bp, "best_val_vcr": study.best_value,
         "test_vcr_mean": float(np.mean(v)), "test_vcr_std": float(np.std(v)),
         "test_mae_mean": float(np.mean([r["mae"] for r in vals])),
         "test_kendall_mean": float(np.mean([r["kendall"] for r in vals]))},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved search_trials.csv / search_best.json")


if __name__ == "__main__":
    main()
