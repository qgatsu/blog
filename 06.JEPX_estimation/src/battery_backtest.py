"""予測価格で蓄電池の充放電を計画し、実績価格で決済するバックテスト。

設備は低圧系統用蓄電所 (50kW未満) を想定:
  出力 49.5 kW / 容量 200 kWh / 往復効率 87% / SOC 10-90%

各日 48 コマを 1 つの LP として解く。日跨ぎの持ち越しは無し (日初=日末 SOC)。
劣化コストは放電量あたりの単価 lambda として目的関数に入れ、感度分析する。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- 設備パラメータ ---
POWER_KW = 49.5          # 出力 (充電・放電とも)
CAPACITY_KWH = 200.0     # 公称容量
SOC_MIN, SOC_MAX = 0.10, 0.90
RT_EFF = 0.87            # 往復効率
DT_H = 0.5               # 1 コマの長さ [h]

E_MIN, E_MAX = CAPACITY_KWH * SOC_MIN, CAPACITY_KWH * SOC_MAX
E_START = E_MIN                      # 日初 SOC = 下限
MAX_PER_SLOT = POWER_KW * DT_H       # 1 コマの最大充放電量 [kWh]
ETA = np.sqrt(RT_EFF)                # 片道効率


def solve_day(prices: np.ndarray, deg_cost: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """1 日 48 コマの充放電計画を LP で解く。prices は円/kWh。

    変数 x = [c_1..c_n, d_1..d_n]  (充電量/放電量 [kWh], いずれも >= 0)
    目的  max  sum p_t*d_t - sum p_t*c_t - deg_cost*sum d_t
    制約  E_MIN <= E_START + cumsum(ETA*c - d/ETA) <= E_MAX  (各 t)
          sum(ETA*c - d/ETA) = 0                             (日初=日末)
    """
    n = len(prices)
    # linprog は最小化なので符号反転
    obj = np.concatenate([prices, -(prices - deg_cost)])

    # SOC 推移の累積和行列
    tri = np.tril(np.ones((n, n)))
    soc_coef = np.hstack([ETA * tri, -tri / ETA])          # 各 t の SOC 増分累積
    A_ub = np.vstack([soc_coef, -soc_coef])
    b_ub = np.concatenate([np.full(n, E_MAX - E_START), np.full(n, E_START - E_MIN)])

    A_eq = np.hstack([ETA * np.ones((1, n)), -np.ones((1, n)) / ETA])
    b_eq = np.zeros(1)

    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, MAX_PER_SLOT)] * (2 * n), method="highs")
    if not res.success:
        return None
    return res.x[:n], res.x[n:]


def settle(charge: np.ndarray, discharge: np.ndarray, actual: np.ndarray,
           deg_cost: float = 0.0) -> dict:
    """計画を実績価格で決済する。"""
    revenue = float(np.sum(actual * discharge))
    cost = float(np.sum(actual * charge))
    thr = float(np.sum(discharge))
    return dict(revenue=revenue, cost=cost, gross=revenue - cost,
                throughput=thr, deg=deg_cost * thr,
                net=revenue - cost - deg_cost * thr)


def run(pred_col: str, df: pd.DataFrame, deg_cost: float = 0.0) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby(df.index.normalize()):
        if len(g) != 48:
            continue
        plan = solve_day(g[pred_col].to_numpy(), deg_cost)
        if plan is None:
            continue
        c, d = plan
        r = settle(c, d, g["target"].to_numpy(), deg_cost)
        r["date"] = date
        r["cycles"] = r["throughput"] / (CAPACITY_KWH * (SOC_MAX - SOC_MIN))
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    pred = pd.read_parquet(DATA_DIR / "predictions_gateclose.parquet")
    feat = pd.read_parquet(DATA_DIR / "features_gateclose.parquet")
    pred["naive"] = feat.loc[pred.index, "lag_7d_same_slot"]
    pred["oracle"] = pred["target"]

    strategies = {"oracle": "oracle", "q50": "pred_q50",
                  "point": "pred_point", "naive": "naive"}

    print(f"設備: {POWER_KW}kW / {CAPACITY_KWH}kWh / 往復効率{RT_EFF:.0%} / "
          f"SOC {SOC_MIN:.0%}-{SOC_MAX:.0%} (実効 {E_MAX-E_MIN:.0f}kWh)")
    print(f"期間: {pred.index.min().date()} ~ {pred.index.max().date()}\n")

    # --- 劣化コスト 0 での比較 ---
    results, daily = {}, {}
    for name, col in strategies.items():
        r = run(col, pred, deg_cost=0.0)
        daily[name] = r
        results[name] = dict(days=len(r), gross_total=r.gross.sum(),
                             gross_per_day=r.gross.mean(),
                             throughput_per_day=r.throughput.mean(),
                             cycles_per_day=r.cycles.mean())

    base = results["oracle"]["gross_total"]
    print("=" * 74)
    print("劣化コスト 0 円/kWh のとき")
    print("=" * 74)
    print(f"{'戦略':10s}{'日数':>5s}{'粗利合計(円)':>14s}{'日次平均(円)':>13s}"
          f"{'オラクル比':>10s}{'サイクル/日':>11s}")
    for name, r in results.items():
        print(f"{name:10s}{r['days']:5d}{r['gross_total']:14,.0f}{r['gross_per_day']:13,.0f}"
              f"{r['gross_total']/base*100:9.1f}%{r['cycles_per_day']:11.2f}")

    # --- 劣化コスト感度 ---
    print("\n" + "=" * 74)
    print("劣化コスト感度 (純利益 = 粗利 - 劣化コスト, 円/日)")
    print("=" * 74)
    lambdas = [0, 5, 10, 15, 17, 20, 25]
    print(f"{'λ(円/kWh)':>10s}" + "".join(f"{n:>12s}" for n in strategies))
    sens = {}
    for lam in lambdas:
        row = {}
        for name, col in strategies.items():
            r = run(col, pred, deg_cost=lam)
            row[name] = r.net.mean()
        sens[lam] = row
        print(f"{lam:>10.0f}" + "".join(f"{row[n]:12,.0f}" for n in strategies))

    # 損益分岐点
    print("\n損益分岐となる劣化コスト (純利益=0 となる λ):")
    for name in strategies:
        lo, hi = 0.0, 60.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if run(name if name == "oracle" else strategies[name], pred, deg_cost=mid).net.mean() > 0:
                lo = mid
            else:
                hi = mid
        print(f"  {name:10s} {lo:5.1f} 円/kWh")

    # 保存
    out = {"setup": dict(power_kw=POWER_KW, capacity_kwh=CAPACITY_KWH, rt_eff=RT_EFF,
                         soc_min=SOC_MIN, soc_max=SOC_MAX,
                         period=[str(pred.index.min()), str(pred.index.max())]),
           "results_no_degradation": results,
           "sensitivity": {str(k): v for k, v in sens.items()}}
    (DATA_DIR / "battery_backtest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.concat({k: v for k, v in daily.items()}, names=["strategy"]).to_parquet(
        DATA_DIR / "battery_daily.parquet")
    print("\nsaved battery_backtest.json / battery_daily.parquet")


if __name__ == "__main__":
    main()
