"""
Q1.5 STL-lite 分解（无 statsmodels）
 分量：
   趋势 T_d = 90 天中心化滚动中位（对 NaN 容忍）
   季节 S_d = 去趋势后按 day-of-year 的平均
   残差 R_d = y_d - T_d - S_d
输出:
  data/stl_components.csv
  figures/fig2_stl_decomposition.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
df = pd.read_csv(ROOT / "data/daily_with_vars.csv", parse_dates=["d"])

def rolling_median_nanaware(y, win=91):
    s = pd.Series(y)
    return s.rolling(window=win, center=True, min_periods=max(15, win//4)).median().values

def doy_mean(detr, doy):
    tab = pd.DataFrame({"doy": doy, "v": detr}).groupby("doy")["v"].mean()
    return tab

parts = []
for i, sub in df.groupby("i"):
    sub = sub.sort_values("d").reset_index(drop=True).copy()
    sub["trend"] = rolling_median_nanaware(sub["y"].values, 91)
    sub["detrended"] = sub["y"] - sub["trend"]
    doy = sub["d"].dt.dayofyear
    seas_table = doy_mean(sub["detrended"].values, doy.values)
    # 居中到 0
    seas_table = seas_table - seas_table.mean()
    sub["season"] = doy.map(seas_table)
    sub["resid"] = sub["y"] - sub["trend"] - sub["season"]
    parts.append(sub)

stl = pd.concat(parts, ignore_index=True)
stl.to_csv(ROOT / "data/stl_components.csv", index=False)

# 作图：每台 4 子图（原始/趋势/季节/残差）
fig, axes = plt.subplots(10, 4, figsize=(18, 24), sharex=True)
for row, i in enumerate(range(1, 11)):
    sub = stl[stl["i"] == i].sort_values("d")
    axes[row, 0].plot(sub["d"], sub["y"], ".", ms=1.5, c="steelblue"); axes[row, 0].set_ylabel(f"A{i}", fontsize=9)
    axes[row, 1].plot(sub["d"], sub["trend"], "-", c="darkorange", lw=1)
    axes[row, 2].plot(sub["d"], sub["season"], "-", c="seagreen", lw=0.7)
    axes[row, 3].plot(sub["d"], sub["resid"], ".", ms=1.2, c="grey")
    for j in range(4): axes[row, j].grid(alpha=0.3)
    if row == 0:
        for j, t in enumerate(["observed y", "trend (91d rolling median)",
                                "seasonal (DOY mean)", "residual"]):
            axes[row, j].set_title(t, fontsize=10)
fig.suptitle("Q1.5  STL-lite decomposition per filter", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures/fig2_stl_decomposition.png", dpi=120)
plt.close(fig)
print(f"Saved: {ROOT/'figures/fig2_stl_decomposition.png'}")

# 指标汇总：线性趋势斜率（稳健）、季节幅度、残差 std
from numpy.polynomial import polynomial as P
summary_rows = []
for i, sub in stl.groupby("i"):
    sub = sub.dropna(subset=["y", "trend"])
    t = sub["t"].values
    tr = sub["trend"].values
    # 以趋势为被解释变量做 OLS 求斜率（天per天）
    mask = ~np.isnan(tr)
    if mask.sum() > 30:
        slope, intercept = np.polyfit(t[mask], tr[mask], 1)
    else:
        slope, intercept = np.nan, np.nan
    season_amp = (sub["season"].max() - sub["season"].min()) if sub["season"].notna().any() else np.nan
    resid_std = sub["resid"].std()
    summary_rows.append(dict(i=i,
                             trend_slope_per_day=slope,
                             trend_slope_per_year=slope*365.25,
                             season_amp=season_amp,
                             resid_std=resid_std))
summary = pd.DataFrame(summary_rows).round(3)
summary.to_csv(ROOT / "data/stl_summary.csv", index=False)
print("STL summary:")
print(summary.to_string(index=False))
