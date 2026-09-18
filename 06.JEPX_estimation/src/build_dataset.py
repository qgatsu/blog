"""JEPX day-ahead 予測データセット構築。

2 系統の特徴量を同一パイプラインで作る:
  - actual   : 気象実測 + 同時刻の需給実績。既存 06 と同じ「事後データ」版。
  - gateclose: スポット入札締切 (前日 10:00) までに実際に入手できる情報だけ。
               気象は前日発行の予報値、需給は D-2 以前の実績ラグ。

情報境界の根拠:
  - JEPX スポットは前日 10:00 締切 / 10:30 頃に約定結果公表。
    => 対象日 D の入札時点 (D-1 10:00) で確定済みの価格は D-1 分まで。
  - 東電エリア需給実績は前日分が翌朝 6 時頃に公開。
    => D-1 10:00 時点で公開済みの実績は D-2 分まで。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import jpholiday
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_JEPX, RAW_TEPCO = DATA_DIR / "raw_jepx", DATA_DIR / "raw_tepco"
RAW_WEATHER, RAW_WEATHER_FC = DATA_DIR / "raw_weather", DATA_DIR / "raw_weather_fc"

TARGET_COL = "area_price_tokyo"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "shortwave_radiation", "precipitation"]
INTERP_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]

# 予報値が利用可能な最初の日 (Open-Meteo previous_runs)
FORECAST_START = pd.Timestamp("2021-04-01")


# ---------------------------------------------------------------- JEPX
JEPX_PATTERNS = {
    "delivery_date": ["年月日", "受渡日"], "slot_code": ["時刻コード"],
    "sell_volume_kwh": ["売り入札量"], "buy_volume_kwh": ["買い入札量"],
    "trade_volume_kwh": ["約定総量"], "system_price": ["システムプライス"],
    "area_price_tokyo": ["エリアプライス東京"],
}


def load_jepx() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_JEPX.glob("spot_*.csv")):
        for enc in ("cp932", "utf-8-sig", "utf-8"):
            try:
                raw = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        mapping = {}
        for new, pats in JEPX_PATTERNS.items():
            for col in raw.columns:
                if any(p in col for p in pats):
                    mapping[col] = new
                    break
        frames.append(raw.rename(columns=mapping)[list(mapping.values())])
    df = pd.concat(frames, ignore_index=True)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["slot_code"] = pd.to_numeric(df["slot_code"], errors="coerce").astype("Int64")
    num = [c for c in df.columns if c not in ("delivery_date", "slot_code")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["delivery_date", "slot_code"]).copy()
    df["datetime"] = df["delivery_date"] + pd.to_timedelta((df["slot_code"].astype(int) - 1) * 30, unit="m")
    return df.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")


# ---------------------------------------------------------------- 気象
def fetch_weather_actual(years: list[int]) -> None:
    """Open-Meteo Archive (実測) を年ごとにキャッシュ取得。"""
    RAW_WEATHER.mkdir(parents=True, exist_ok=True)
    for year in years:
        out = RAW_WEATHER / f"tokyo_{year}.json"
        end = min(date(year, 12, 31), date.today() - timedelta(days=1))
        if out.exists():
            cached = json.loads(out.read_text())["hourly"]["time"][-1][:10]
            if pd.Timestamp(cached).date() >= end - timedelta(days=2):
                continue
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": 35.6895, "longitude": 139.6917,
            "start_date": f"{year}-01-01", "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_VARS), "timezone": "Asia/Tokyo"}, timeout=120)
        r.raise_for_status()
        out.write_text(r.text, encoding="utf-8")
        print(f"  [weather actual] {year} updated -> {end}")


def _load_hourly(dir_: Path, pattern: str, cols: list[str], rename: dict | None = None) -> pd.DataFrame:
    frames = []
    for p in sorted(dir_.glob(pattern)):
        h = json.loads(p.read_text(encoding="utf-8"))["hourly"]
        df = pd.DataFrame({c: h[c] for c in ["time"] + cols})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        frames.append(df.apply(pd.to_numeric, errors="coerce"))
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.rename(columns=rename) if rename else out


def resample_weather(hourly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """1h -> 30min。連続変数は時間補間、降水は ffill (積算量のため)。"""
    interp = hourly[[c for c in INTERP_VARS if c in hourly]].resample("30min").interpolate(method="time")
    rain = hourly[["precipitation"]].resample("30min").ffill()
    out = pd.concat([interp, rain], axis=1).reindex(index)
    out[interp.columns] = out[interp.columns].ffill().bfill()
    out["precipitation"] = out["precipitation"].fillna(0.0)
    return out[WEATHER_VARS]


# ---------------------------------------------------------------- 東電需給
TEPCO_OLD = ["DATE", "TIME", "demand", "nuclear", "thermal", "hydro", "geothermal", "biomass",
             "solar", "solar_curtail", "wind", "wind_curtail", "pumped", "interconnect", "total"]
TEPCO_NEW = ["DATE", "TIME", "demand", "nuclear", "thermal_lng", "thermal_coal", "thermal_oil",
             "thermal_other", "hydro", "geothermal", "biomass", "solar", "solar_curtail",
             "wind", "wind_curtail", "pumped", "battery", "interconnect", "other", "total"]


def load_tepco() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_TEPCO.glob("area-*.csv")):          # 年度別 1h, 万kWh
        df = pd.read_csv(path, encoding="cp932", skiprows=3, names=TEPCO_OLD, header=None)
        df["datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        frames.append(pd.DataFrame({
            "tepco_demand": pd.to_numeric(df["demand"], errors="coerce") * 10,
            "tepco_solar": pd.to_numeric(df["solar"], errors="coerce") * 10,
            "tepco_wind": pd.to_numeric(df["wind"], errors="coerce") * 10}))
    for path in sorted(RAW_TEPCO.glob("eria_jukyu_*.csv")):    # 月次 30min, MW
        df = pd.read_csv(path, encoding="utf-8", skiprows=2, names=TEPCO_NEW, header=None)
        df["datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        frames.append(df[["demand", "solar", "wind"]].apply(pd.to_numeric, errors="coerce")
                      .rename(columns={"demand": "tepco_demand", "solar": "tepco_solar", "wind": "tepco_wind"}))
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def resample_tepco(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    out = raw.resample("30min").interpolate(method="time").reindex(index).ffill().bfill()
    out["net_demand"] = out["tepco_demand"] - out["tepco_solar"] - out["tepco_wind"]
    out["pv_ratio"] = (out["tepco_solar"] / out["tepco_demand"]).clip(0, 1)
    return out


# ---------------------------------------------------------------- 特徴量
def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour.astype("int16")
    df["minute"] = index.minute.astype("int16")
    df["dayofweek"] = index.dayofweek.astype("int16")
    df["month"] = index.month.astype("int16")
    df["day"] = index.day.astype("int16")
    df["is_weekend"] = (index.dayofweek >= 5).astype("int8")
    dates = pd.Index(index.normalize().unique())
    hmap = {d: int(jpholiday.is_holiday(d.date())) for d in dates}
    df["is_holiday"] = index.normalize().map(hmap).astype("int8")
    df["is_dayoff"] = ((df.is_weekend == 1) | (df.is_holiday == 1)).astype("int8")
    return df


def build_price_lags(price: pd.Series) -> pd.DataFrame:
    """価格ラグ。D-1 分は D-2 10:30 に確定済みなので入札時点で既知。"""
    df = pd.DataFrame(index=price.index)
    df["lag_1d_same_slot"] = price.shift(48)
    df["lag_7d_same_slot"] = price.shift(48 * 7)
    dates = price.index.normalize()
    daily = price.groupby(dates).agg(["mean", "max", "min", "std"])
    for shift, prefix in ((1, "prev_day"), (7, "prev_week_same_dow")):
        s = daily.shift(shift)
        s.columns = [f"{prefix}_{c}" for c in s.columns]
        df = df.join(s.reindex(dates).set_axis(price.index))
    hour = price.index.hour
    band = pd.DataFrame({"price": price.to_numpy(), "date": dates})
    band["peak"] = np.where((hour >= 17) & (hour < 20), band.price, np.nan)
    band["early"] = np.where((hour >= 3) & (hour < 6), band.price, np.nan)
    bd = band.groupby("date")[["peak", "early"]].mean().shift(1)
    bd.columns = [f"prev_day_band_{c}" for c in bd.columns]
    return df.join(bd.reindex(dates).set_axis(price.index))


def build_supply_lags(tepco: pd.DataFrame) -> pd.DataFrame:
    """需給実績のラグ。前日 10:00 時点で公開済みなのは D-2 分までなので 2 日以上前に限る。"""
    df = pd.DataFrame(index=tepco.index)
    for col in ["tepco_demand", "tepco_solar", "net_demand"]:
        df[f"{col}_lag2d"] = tepco[col].shift(48 * 2)
        df[f"{col}_lag7d"] = tepco[col].shift(48 * 7)
    dates = tepco.index.normalize()
    daily = tepco[["tepco_demand", "tepco_solar", "net_demand"]].groupby(dates).mean().shift(2)
    daily.columns = [f"{c}_prev2d_mean" for c in daily.columns]
    return df.join(daily.reindex(dates).set_axis(tepco.index))


def add_target_variants(df: pd.DataFrame, full_price: pd.Series) -> tuple[pd.DataFrame, float]:
    offset = max(0.0, -float(df["target"].min())) + 1.0
    df["target_log1p"] = np.log1p(df["target"] + offset)
    df["target_arsinh"] = np.arcsinh(df["target"])
    df["target_diff_1d"] = df["target"] - df["lag_1d_same_slot"]
    df["target_diff_7d"] = df["target"] - df["lag_7d_same_slot"]
    df["target_dev_prev_day"] = df["target"] - df["prev_day_mean"]
    df["target_dev_prev_week"] = df["target"] - df["prev_week_same_dow_mean"]
    daily_mean = full_price.groupby(full_price.index.normalize()).mean()
    trend = daily_mean.shift(1).rolling(30, min_periods=5).mean()
    df["trend_30d_mean"] = pd.Series(df.index.normalize().map(trend).to_numpy(), index=df.index)
    return df, offset


# ---------------------------------------------------------------- main
def main() -> None:
    print("[1/5] JEPX")
    jepx = load_jepx()
    price = jepx[TARGET_COL]
    index = jepx.index
    print(f"      {len(jepx):,} rows  {index.min()} ~ {index.max()}")
    jepx.to_parquet(DATA_DIR / "jepx_clean.parquet")

    print("[2/5] 気象 (ERA5 / 解析値相当 / 前日予報)")
    fetch_weather_actual(sorted({d.year for d in index}))
    # (a) ERA5 再解析 = 既存 06 が使っていた実測
    w_era5 = resample_weather(_load_hourly(RAW_WEATHER, "tokyo_*.json", WEATHER_VARS), index)
    # (b) previous_runs の最新ラン = 予報と同一モデル系列の「事後に分かる値」
    w_analysis = resample_weather(_load_hourly(RAW_WEATHER_FC, "tokyo_fc_*.json", WEATHER_VARS), index)
    # (c) 入札時点で入手できる予報。
    #     previous_dayN は「対象時刻の N*24 時間前に発行された予報」という固定リードタイム。
    #     スポットの入札締切は対象日 D の前日 10:00 なので、day1 (24h前発行) だと
    #     D 10:30 以降のコマで締切後に発行された予報を使うことになりリークする。
    #     day2 (48h前発行) なら D 23:30 のコマでも D-2 23:30 発行となり全コマで安全。
    fc_cols = [f"{v}_previous_day2" for v in WEATHER_VARS]
    w_fc = resample_weather(_load_hourly(RAW_WEATHER_FC, "tokyo_fc_*.json", fc_cols,
                                         rename=dict(zip(fc_cols, WEATHER_VARS))), index)
    for w in (w_analysis, w_fc):
        w.loc[w.index < FORECAST_START] = np.nan       # 予報系が存在しない期間は欠損
    w_era5.to_parquet(DATA_DIR / "weather_tokyo_30min.parquet")
    w_analysis.to_parquet(DATA_DIR / "weather_tokyo_30min_analysis.parquet")
    w_fc.to_parquet(DATA_DIR / "weather_tokyo_30min_forecast.parquet")

    print("[3/5] 東電需給")
    tepco = resample_tepco(load_tepco(), index)
    tepco.to_parquet(DATA_DIR / "tepco_supply_30min.parquet")

    print("[4/5] 特徴量")
    cal = build_calendar(index)
    price_lags = build_price_lags(price)
    supply_lags = build_supply_lags(tepco)
    target = price.rename("target").to_frame()

    # 気象ソースと需給情報を 2x2 で切り分け、どの制約がどれだけ効くかを分離する。
    variants = {
        # (1) 既存 06 と同じ形。ERA5 実測 + 同時刻の需給実績
        "era5_actual": target.join(cal).join(w_era5).join(tepco).join(price_lags),
        # (2) 気象を予報と同一モデル系列の事後値に。(1) との差はデータソース差のみ
        "analysis": target.join(cal).join(w_analysis).join(tepco).join(price_lags),
        # (3) 気象だけ前日予報に。(2) との差が純粋な気象予報誤差の影響
        "fc_weather": target.join(cal).join(w_fc).join(tepco).join(price_lags),
        # (4) 需給も D-2 以前に。(3) との差が需給情報の制約の影響。これが実運用の条件
        "gateclose": target.join(cal).join(w_fc).join(supply_lags).join(price_lags),
    }

    meta_all = {}
    for name, df in variants.items():
        need = ["target"] + list(price_lags.columns)
        if name in ("analysis", "fc_weather", "gateclose"):
            need += list(WEATHER_VARS)
        if name == "gateclose":
            need += list(supply_lags.columns)
        before = len(df)
        df = df.dropna(subset=need)
        df, offset = add_target_variants(df, price)
        out = DATA_DIR / f"features_{name}.parquet"
        df.to_parquet(out)
        meta_all[name] = {
            "rows": len(df), "dropped": before - len(df),
            "period": [str(df.index.min()), str(df.index.max())],
            "columns": list(df.columns), "log1p_offset": offset,
        }
        print(f"      {name:10s} {len(df):,} rows ({before - len(df):,} dropped)  "
              f"{df.index.min().date()} ~ {df.index.max().date()}  cols={len(df.columns)}")

    print("[5/5] メタ情報")
    meta_all["_note"] = {
        "gate_closure": "JEPX スポット入札締切 = 前日 10:00 / 約定結果公表 10:30 頃",
        "price_lag_policy": "D-1 の価格は D-2 10:30 に確定済みのため入札時点で既知",
        "supply_lag_policy": "東電需給実績は前日分が翌朝 6 時公開のため D-2 以前のみ使用",
        "weather_policy": {
            "era5_actual": "Open-Meteo Archive (ERA5 再解析) の実測値 + 同時刻の需給実績",
            "analysis": "previous_runs の最新ラン (予報と同一モデル系列の事後値) + 同時刻の需給実績",
            "fc_weather": "previous_runs の 48 時間前発行ラン + 同時刻の需給実績",
            "gateclose": "previous_runs の 48 時間前発行ラン + D-2 以前の需給実績ラグ",
        },
        "comparison_design": {
            "era5_actual vs analysis": "気象データソースの差 (ERA5 vs 予報モデル)",
            "analysis vs fc_weather": "気象予報誤差の影響 (48時間前発行の予報)",
            "fc_weather vs gateclose": "需給情報が D-2 までに限られることの影響",
        },
        "forecast_available_from": str(FORECAST_START.date()),
        "forecast_lead_time": "previous_day2 = 対象時刻の48時間前に発行された予報。"
                              "入札締切 (前日10:00) より前に確実に入手できる最も近いランとして選択。"
                              "day1 (24時間前発行) は D 10:30 以降のコマでリークする。",
    }
    (DATA_DIR / "dataset_meta.json").write_text(
        json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print("      saved dataset_meta.json")


if __name__ == "__main__":
    main()
