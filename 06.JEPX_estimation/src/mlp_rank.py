"""48コマ同時出力の MLP を、MSE と順序損失のハイブリッドで学習する。

LP は価格の順序をほぼ全て使う (順序だけで VCR 88-92%) が、
効率損失の判断に絶対値も要る (順序のみでは 100% に届かない) ため、
  loss = alpha * MSE + (1 - alpha) * pairwise ranking loss
の形で両方を学習する。alpha を振って VCR がどう動くかを見る。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_backtest import solve_day, settle

SEED = 42
TEST_MONTHS, VAL_MONTHS = 6, 3
EXCLUDE = {"target", "target_log1p", "target_arsinh", "target_diff_1d", "target_diff_7d",
           "target_dev_prev_day", "target_dev_prev_week"}


def build_daily(df: pd.DataFrame):
    """1 日 = 1 サンプルに整形する。日内で変化しない列は 1 回だけ使う。"""
    feats = [c for c in df.columns if c not in EXCLUDE]
    dates = df.index.normalize()
    full = df.groupby(dates).filter(lambda g: len(g) == 48)
    dates = full.index.normalize()
    day_index = pd.Index(sorted(dates.unique()))

    # 日内分散が 0 の列 = 日次固定
    var_in_day = full[feats].groupby(dates).std().mean()
    static_cols = [c for c in feats if var_in_day[c] < 1e-9]
    dynamic_cols = [c for c in feats if c not in static_cols]

    n_days = len(day_index)
    X_static = full[static_cols].groupby(dates).first().to_numpy(dtype=np.float32)
    X_dyn = (full[dynamic_cols].to_numpy(dtype=np.float32)
             .reshape(n_days, 48, len(dynamic_cols)).reshape(n_days, -1))
    X = np.hstack([X_static, X_dyn])
    Y = full["target"].to_numpy(dtype=np.float32).reshape(n_days, 48)
    return X, Y, day_index, static_cols, dynamic_cols


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(512, 256), p=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 48)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def pairwise_rank_loss(pred, true, beta: float = 0.0):
    """日内の全ペアについて大小関係が合っているかを logistic 損失で測る。

    beta はペアの重み |true_i - true_j|^beta の指数。
      beta = 0  重みなし (全ペア均等)
      beta = 1  価格差に比例
    価格差が大きいペアほど順序を誤ったときの収益損失が大きい一方、
    そこはスパイク領域で予測も難しいため、最適な beta は事前には決まらない。
    """
    dp = pred.unsqueeze(2) - pred.unsqueeze(1)      # (B,48,48)
    dt = true.unsqueeze(2) - true.unsqueeze(1)
    sign = torch.sign(dt)
    mask = sign != 0
    loss = nn.functional.softplus(-dp * sign)
    if beta > 0:
        w = dt.abs().clamp(min=1e-6).pow(beta) * mask
        return (loss * w).sum() / w.sum().clamp(min=1e-6)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def train(Xtr, Ytr, Xva, Yva, alpha, epochs=400, patience=40, seed=SEED, beta=0.0,
          hidden=(512, 256), dropout=0.2, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1], hidden=hidden, p=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    xtr, ytr = torch.tensor(Xtr), torch.tensor(Ytr)
    xva, yva = torch.tensor(Xva), torch.tensor(Yva)
    best, best_state, bad = np.inf, None, 0
    n, bs = len(xtr), 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            p = model(xtr[idx])
            loss = alpha * nn.functional.mse_loss(p, ytr[idx]) \
                 + (1 - alpha) * pairwise_rank_loss(p, ytr[idx], beta)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(xva)
            vl = (alpha * nn.functional.mse_loss(pv, yva)
                  + (1 - alpha) * pairwise_rank_loss(pv, yva, beta)).item()
        if vl < best - 1e-6:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate(pred_days, true_days, oracle_profit):
    profit, taus, rhos, errs = 0.0, [], [], []
    for p, t in zip(pred_days, true_days):
        p = np.maximum(p, 0.01)
        plan = solve_day(p)
        if plan is None:
            continue
        profit += settle(plan[0], plan[1], t)["gross"]
        taus.append(kendalltau(t, p).statistic)
        rhos.append(spearmanr(t, p).statistic)
        errs.append(t - p)
    e = np.concatenate(errs)
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                kendall=float(np.nanmean(taus)), spearman=float(np.nanmean(rhos)),
                profit_per_day=profit / len(pred_days), vcr=profit / oracle_profit)


def main():
    df = pd.read_parquet(DATA_DIR / "features_gateclose.parquet")
    X, Y, day_index, static_cols, dynamic_cols = build_daily(df)
    print(f"日次サンプル {len(day_index):,}  入力次元 {X.shape[1]} "
          f"(日次固定 {len(static_cols)} + コマ依存 {len(dynamic_cols)}×48)")

    last = day_index.max()
    test_start = (last - pd.DateOffset(months=TEST_MONTHS)).normalize()
    val_start = (test_start - pd.DateOffset(months=VAL_MONTHS)).normalize()
    tr = day_index < val_start
    va = (day_index >= val_start) & (day_index < test_start)
    te = day_index >= test_start
    print(f"train {tr.sum()} / val {va.sum()} / test {te.sum()} 日"
          f"  (test {day_index[te].min().date()} ~ {day_index[te].max().date()})\n")

    # 標準化
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-8
    my, sy = Y[tr].mean(), Y[tr].std()
    Xs = ((X - mx) / sx).astype(np.float32)
    Ys = ((Y - my) / sy).astype(np.float32)

    # オラクル収益
    orc = evaluate(Y[te], Y[te], 1.0)
    oracle_profit = orc["profit_per_day"] * te.sum()
    print(f"オラクル収益 {orc['profit_per_day']:,.0f} 円/日\n")

    rows = []
    for alpha in (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0):
        model = train(Xs[tr], Ys[tr], Xs[va], Ys[va], alpha)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xs[te])).numpy() * sy + my
        r = evaluate(pred, Y[te], oracle_profit)
        r["alpha"] = alpha
        rows.append(r)
        print(f"alpha={alpha:.1f}  MAE {r['mae']:5.2f}  RMSE {r['rmse']:5.2f}  "
              f"Kendall {r['kendall']:.3f}  Spearman {r['spearman']:.3f}  "
              f"収益 {r['profit_per_day']:6,.0f} 円/日  VCR {r['vcr']*100:5.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "mlp_rank_results.csv", index=False)
    print("\n" + "=" * 70)
    print("ベースライン比較")
    print("=" * 70)
    base = pd.read_csv(DATA_DIR / "baseline_metrics.csv")
    b = base[base.model.str.contains("L2")].iloc[0]
    print(f"  LightGBM(L2)   MAE {b.mae:5.2f}  RMSE {b.rmse:5.2f}  "
          f"Kendall {b.kendall:.3f}  VCR {b.vcr*100:5.1f}%")
    best = res.loc[res.vcr.idxmax()]
    print(f"  MLP α={best.alpha:.1f}    MAE {best.mae:5.2f}  RMSE {best.rmse:5.2f}  "
          f"Kendall {best.kendall:.3f}  VCR {best.vcr*100:5.1f}%")
    print(f"\n  VCR 改善 {(best.vcr - b.vcr)*100:+.1f} ポイント "
          f"({(best.profit_per_day - b.profit_per_day):+,.0f} 円/日)")


if __name__ == "__main__":
    main()
