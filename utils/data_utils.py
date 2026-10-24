"""Data prep + forecasting for the 3D COVID-19 dashboard (no Prophet needed)."""
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"
SKIP = ("Diamond Princess", "MS Zaandam", "Olympics")

def _clean(name):
    return name.split(".")[0].split("(")[0].strip()

def _load(fname):
    df = pd.read_csv(D / fname, encoding="latin-1").iloc[1:]          # row 0 = "Province/State"
    df = df.set_index("Country/Region").apply(pd.to_numeric, errors="coerce").fillna(0)
    df.index = pd.to_datetime(df.index, format="%m/%d/%y")
    return df.clip(lower=0)                                            # daily NEW counts

def forecast(daily, h=30):
    """Damped log-trend on the 7-day mean x weekly seasonality; ~80% band from recent volatility."""
    y = pd.Series(daily, dtype=float)
    sm = y.rolling(7).mean()
    ratio = (y / sm.shift(-3)).tail(59).head(56).reset_index(drop=True)   # centered 7-day mean
    dow = ratio.groupby(np.arange(56) % 7).mean().values
    dow = dow / dow.mean()
    ls = np.log1p(sm.dropna().values)
    slope = float(np.clip(np.polyfit(np.arange(28), ls[-28:], 1)[0], -0.06, 0.06))
    sigma = max(float(np.std(np.diff(ls[-60:]))), 0.02)
    k = np.arange(1, h + 1)
    lvl = ls[-1] + slope * np.cumsum(0.95 ** k)
    start = len(y) - 59                                                    # index of ratio[0]
    seas = dow[(np.arange(len(y), len(y) + h) - start) % 7]
    mid = np.expm1(lvl) * seas
    w = 1.28 * sigma * np.sqrt(k)
    return mid, mid * np.exp(-w), mid * np.exp(w)

def backtest_mape(daily, h=30):
    mid, _, _ = forecast(daily[:-h], h)
    a = np.asarray(daily[-h:], dtype=float)
    return float(np.mean(np.abs(mid - a) / np.maximum(a, 1)) * 100)

def build_payload():
    cases, deaths = _load("confirmed.csv"), _load("deaths.csv")
    meta = pd.read_csv(D / "metadata.csv", encoding="latin-1")
    meta.index = cases.columns                                           # same column order
    meta["c"] = [_clean(n) for n in meta["Country/Region"]]
    meta = meta[~meta["c"].str.contains("|".join(SKIP))].dropna(subset=["Lat", "Long"])
    main = meta[meta["Province/State"].isna()]                           # mainland row wins over territories
    rest = meta[~meta["c"].isin(main["c"])]
    geo = pd.concat([main, rest]).groupby("c")[["Lat", "Long"]].mean()
    names = [c for c in geo.index if (geo.loc[c] != 0).any()]
    col2c = meta["c"].to_dict()

    dates = cases.index
    idx = list(range(0, len(dates), 7)) + [len(dates) - 1]
    out = {"dates": [d.strftime("%d %b %Y") for d in dates[idx]], "countries": []}
    series = {}
    for key, df in (("cases", cases), ("deaths", deaths)):
        g = df[[c for c in df.columns if c in col2c]].T.groupby(col2c).sum().T
        series[key] = g
    for c in names:
        out["countries"].append({
            "n": c, "lat": round(float(geo.loc[c, "Lat"]), 2), "lon": round(float(geo.loc[c, "Long"]), 2),
            "cases": [int(v) for v in series["cases"][c].cumsum().iloc[idx]],
            "deaths": [int(v) for v in series["deaths"][c].cumsum().iloc[idx]]})
    out["forecast"] = {}
    for key, df in (("cases", cases), ("deaths", deaths)):
        daily = df.sum(axis=1).values
        mid, lo, hi = forecast(daily)
        hist = pd.Series(daily).rolling(7).mean().dropna().tail(150).round().astype(int).tolist()
        fd = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30)
        hd = dates[-len(hist):]
        out["forecast"][key] = {"hist": hist, "mid": mid.round().astype(int).tolist(),
            "lo": lo.round().astype(int).tolist(), "hi": hi.round().astype(int).tolist(),
            "hd": [d.strftime("%d %b %y") for d in hd], "fd": [d.strftime("%d %b %y") for d in fd],
            "mape": round(backtest_mape(daily), 1)}
    return out
