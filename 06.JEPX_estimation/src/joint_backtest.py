"""スポット市場と一次調整力市場への同時参加を最適化するバックテスト。

意思決定の時間構造:
  前日 10:00  JEPX スポット入札締切   <- この時点で調整力価格は未確定
  前日 10:30  スポット約定結果公表
  前日 14:00  一次調整力 入札締切
  前日 15:00  一次調整力 約定

したがってスポット入札時には調整力価格の見込みが要る。ここでは A 案として
「前日同コマの約定価格」を調整力価格の予測に使う (前日分は D-2 15時に約定済み)。

LP の変数 (各コマ t, n=48):
  pc_t 充電電力 [kW], pd_t 放電電力 [kW], r_t 上げ調整力提供量 [kW]
目的:
  max  sum p_t*(pd_t - pc_t)*dt  +  sum q_t*r_t      (q は 円/kW・30分)
制約:
  上げ余力     r_t <= P - pd_t + pc_t
  SOC 範囲     E_MIN <= s_t <= E_MAX
  調整力の裏付け s_t - r_t*(RESP_MIN/60)/eta >= E_MIN
  日初=日末    s_n = s_0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

POWER_KW = 49.5
CAPACITY_KWH = 200.0
SOC_MIN, SOC_MAX = 0.10, 0.90
RT_EFF = 0.87
DT_H = 0.5
RESP_MIN = 5.0            # 一次調整力の継続時間要件 [分]

E_MIN, E_MAX = CAPACITY_KWH * SOC_MIN, CAPACITY_KWH * SOC_MAX
E_START = CAPACITY_KWH * 0.5      # 日初 SOC は中間 (調整力の裏付けを持てるように)
ETA = np.sqrt(RT_EFF)
RESERVE_KWH_PER_KW = (RESP_MIN / 60.0) / ETA     # 1kW 提供するのに要る SOC 余裕


def solve_day(spot: np.ndarray, reserve: np.ndarray | None,
              deg_cost: float = 0.0, allow_spot: bool = True) -> dict | None:
    """1 日 48 コマを LP で解く。reserve=None なら調整力に参加しない。"""
    n = len(spot)
    use_reserve = reserve is not None
    nv = 3 * n if use_reserve else 2 * n

    # 目的 (linprog は最小化なので符号反転)
    obj = np.zeros(nv)
    obj[:n] = spot * DT_H                              # 充電: 支払い
    obj[n:2 * n] = -(spot - deg_cost) * DT_H           # 放電: 受取 - 劣化
    if use_reserve:
        obj[2 * n:] = -reserve                         # 調整力: 受取 [円/kW]

    tri = np.tril(np.ones((n, n)))
    # SOC 増分の累積 [kWh]
    soc = np.zeros((n, nv))
    soc[:, :n] = ETA * tri * DT_H
    soc[:, n:2 * n] = -tri / ETA * DT_H

    A_ub = [soc, -soc]
    b_ub = [np.full(n, E_MAX - E_START), np.full(n, E_START - E_MIN)]

    if use_reserve:
        # 上げ余力: r_t + pd_t - pc_t <= P
        cap = np.zeros((n, nv))
        cap[:, :n] = -np.eye(n)
        cap[:, n:2 * n] = np.eye(n)
        cap[:, 2 * n:] = np.eye(n)
        A_ub.append(cap)
        b_ub.append(np.full(n, POWER_KW))
        # 調整力の裏付け: -soc_t + r_t*RESERVE_KWH_PER_KW <= E_START - E_MIN
        back = -soc.copy()
        back[:, 2 * n:] = np.eye(n) * RESERVE_KWH_PER_KW
        A_ub.append(back)
        b_ub.append(np.full(n, E_START - E_MIN))

    A_eq = np.zeros((1, nv))
    A_eq[0, :n] = ETA * DT_H
    A_eq[0, n:2 * n] = -DT_H / ETA

    bounds = [(0, POWER_KW if allow_spot else 0)] * (2 * n)
    if use_reserve:
        bounds += [(0, POWER_KW)] * n

    res = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=np.zeros(1), bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x
    return dict(pc=x[:n], pd=x[n:2 * n], r=x[2 * n:] if use_reserve else np.zeros(n))


def settle(plan: dict, spot_actual: np.ndarray, reserve_actual: np.ndarray,
           deg_cost: float = 0.0, fill_rate: float = 1.0) -> dict:
    """計画を実績価格で決済する。調整力は容量対価のみ (指令による電力量は扱わない)。

    fill_rate: 応札した調整力のうち実際に落札される比率。1.0 は全量落札の理想値。
    """
    charge_kwh = plan["pc"] * DT_H
    discharge_kwh = plan["pd"] * DT_H
    spot_rev = float(np.sum(spot_actual * discharge_kwh))
    spot_cost = float(np.sum(spot_actual * charge_kwh))
    reserve_rev = float(np.sum(reserve_actual * plan["r"])) * fill_rate
    thr = float(np.sum(discharge_kwh))
    # 調整力として提供した容量は、周波数変動に応じて常時応動する。
    # 提供 kW x 提供時間 x 応動率 を追加スループットとして数える (上下は相殺され、
    # 絶対値だけが電池を消耗させると仮定)。
    reserve_kwh = float(np.sum(plan["r"])) * DT_H * fill_rate
    return dict(spot_gross=spot_rev - spot_cost, reserve_rev=reserve_rev,
                gross=spot_rev - spot_cost + reserve_rev,
                throughput=thr, reserve_capacity_kwh=reserve_kwh,
                net=spot_rev - spot_cost + reserve_rev - deg_cost * thr,
                reserve_kw_mean=float(np.mean(plan["r"])))


def main() -> None:
    spot = pd.read_parquet(DATA_DIR / "predictions_gateclose.parquet")
    res = pd.read_parquet(DATA_DIR / "reserve_primary_tokyo.parquet")
    df = spot.join(res[["reserve_price"]], how="inner").dropna(subset=["reserve_price"])
    # A 案: 調整力価格の予測 = 前日同コマの約定価格
    df["reserve_pred"] = df["reserve_price"].shift(48)
    df = df.dropna(subset=["reserve_pred"])

    print(f"評価期間: {df.index.min().date()} ~ {df.index.max().date()}  {len(df):,} コマ")
    print(f"設備: {POWER_KW}kW / {CAPACITY_KWH}kWh / 効率{RT_EFF:.0%} / SOC {SOC_MIN:.0%}-{SOC_MAX:.0%}")
    print(f"調整力: 継続{RESP_MIN:.0f}分 → 1kW 提供に SOC {RESERVE_KWH_PER_KW:.3f} kWh 必要\n")

    strategies = {
        "spot_only":     dict(spot_col="pred_q50", reserve=False, allow_spot=True),
        "reserve_only":  dict(spot_col="pred_q50", reserve=True,  allow_spot=False),
        "joint":         dict(spot_col="pred_q50", reserve=True,  allow_spot=True),
        "joint_oracle":  dict(spot_col="target",   reserve=True,  allow_spot=True, oracle=True),
    }

    out, daily = {}, {}
    for name, cfg in strategies.items():
        rows = []
        for date, g in df.groupby(df.index.normalize()):
            if len(g) != 48:
                continue
            rp = (g["reserve_price"] if cfg.get("oracle") else g["reserve_pred"]).to_numpy() \
                 if cfg["reserve"] else None
            plan = solve_day(g[cfg["spot_col"]].to_numpy(), rp, allow_spot=cfg["allow_spot"])
            if plan is None:
                continue
            r = settle(plan, g["target"].to_numpy(), g["reserve_price"].to_numpy())
            r["date"] = date
            rows.append(r)
        d = pd.DataFrame(rows).set_index("date")
        daily[name] = d
        out[name] = dict(days=len(d), spot=d.spot_gross.mean(), reserve=d.reserve_rev.mean(),
                         total=d.gross.mean(), throughput=d.throughput.mean(),
                         reserve_kw=d.reserve_kw_mean.mean())

    print("=" * 80)
    print("日次平均収益 (円/日, 劣化コスト前)")
    print("=" * 80)
    print(f"{'戦略':14s}{'日数':>5s}{'スポット':>11s}{'調整力':>11s}{'合計':>11s}"
          f"{'調整力平均kW':>13s}{'放電量kWh/日':>13s}")
    for name, r in out.items():
        print(f"{name:14s}{r['days']:5d}{r['spot']:11,.0f}{r['reserve']:11,.0f}{r['total']:11,.0f}"
              f"{r['reserve_kw']:13.1f}{r['throughput']:13.1f}")

    base = out["spot_only"]["total"]
    print(f"\nスポット単独比:")
    for name, r in out.items():
        print(f"  {name:14s} {r['total']/base:5.2f} 倍   年換算 {r['total']*365/1e4:6,.0f} 万円")

    # 劣化コスト感度
    print("\n" + "=" * 80)
    print("劣化コスト感度 (純利益 円/日)")
    print("=" * 80)
    print(f"{'λ(円/kWh)':>10s}" + "".join(f"{n:>14s}" for n in strategies))
    for lam in (0, 10, 17, 25):
        line = f"{lam:>10.0f}"
        for name in strategies:
            d = daily[name]
            line += f"{(d.gross - lam * d.throughput).mean():14,.0f}"
        print(line)

    # 落札率の感度 (joint 戦略)
    print("\n" + "=" * 80)
    print("落札率の感度 (joint 戦略, 劣化コスト 17円/kWh, 円/日)")
    print("=" * 80)
    g = daily["joint"]
    print(f"{'落札率':>8s}{'調整力収益':>13s}{'合計(粗)':>12s}{'純利益':>12s}")
    for fr in (1.0, 0.8, 0.64, 0.5, 0.3):
        rev = g.reserve_rev.mean() * fr
        gross = g.spot_gross.mean() + rev
        print(f"{fr:8.2f}{rev:13,.0f}{gross:12,.0f}{gross - 17 * g.throughput.mean():12,.0f}")

    # 調整力の応動率 u の感度: 総スループット = 放電量 + 提供容量kWh x u
    print("\n" + "=" * 80)
    print("調整力の応動率 u の感度 (純利益 円/日, 劣化 17円/kWh, 落札率 1.0)")
    print("  総スループット = スポット放電量 + 調整力提供容量[kWh] x u")
    print("=" * 80)
    LAM = 17.0
    print(f"{'u':>7s}" + "".join(f"{n:>14s}" for n in strategies))
    for u in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        line = f"{u:7.0%}"
        for name in strategies:
            d = daily[name]
            thr = d.throughput + d.reserve_capacity_kwh * u
            line += f"{(d.gross - LAM * thr).mean():14,.0f}"
        print(line)

    print("\n結論が逆転する応動率:")
    for a, b in (("joint", "reserve_only"), ("joint", "spot_only")):
        lo, hi = 0.0, 3.0
        da, db = daily[a], daily[b]
        f = lambda u, d: (d.gross - LAM * (d.throughput + d.reserve_capacity_kwh * u)).mean()
        if (f(lo, da) > f(lo, db)) == (f(hi, da) > f(hi, db)):
            print(f"  {a} vs {b}: u=0-300% の範囲で逆転しない")
            continue
        for _ in range(40):
            mid = (lo + hi) / 2
            if (f(mid, da) > f(mid, db)) == (f(lo, da) > f(lo, db)):
                lo = mid
            else:
                hi = mid
        print(f"  {a} vs {b}: u = {lo:.1%} で逆転")

    (DATA_DIR / "joint_backtest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.concat(daily, names=["strategy"]).to_parquet(DATA_DIR / "joint_daily.parquet")
    print("\nsaved joint_backtest.json / joint_daily.parquet")


if __name__ == "__main__":
    main()
