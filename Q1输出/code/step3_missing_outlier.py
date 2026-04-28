"""
Q1.3 缺失与异常检测
 - 连续缺测段统计
 - 基于 IQR*1.5 与 3*MAD 两版异常检测（稳健）
 - 作图：每台的时间序列 + 异常点标记 + 维护事件竖线
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
daily = pd.read_csv(ROOT / "data/daily_long.csv", parse_dates=["d"])

# ---------- 连续缺测段 ----------
def longest_gap(sub):
    # sub 按时间升序；返回最长连续 y=NaN 的长度
    isnan = sub["y"].isna().values
    best = cur = 0
    for x in isnan:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best

gap = (daily.sort_values(["i", "d"])
            .groupby("i").apply(longest_gap, include_groups=False)
            .rename("longest_gap_days"))
print("Longest consecutive NaN gap per filter:")
print(gap.to_string())

# ---------- 异常检测（IQR 1.5 与 3×MAD 两版） ----------
def detect_outlier(sub):
    y = sub["y"].dropna()
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo1, hi1 = q1 - 1.5*iqr, q3 + 1.5*iqr
    med = y.median()
    mad = (y - med).abs().median()
    lo2, hi2 = med - 3*1.4826*mad, med + 3*1.4826*mad
    return pd.Series(dict(lo_iqr=lo1, hi_iqr=hi1, lo_mad=lo2, hi_mad=hi2,
                          n_outlier_iqr=((sub["y"]<lo1)|(sub["y"]>hi1)).sum(),
                          n_outlier_mad=((sub["y"]<lo2)|(sub["y"]>hi2)).sum()))

thr = daily.groupby("i").apply(detect_outlier, include_groups=False)
print("\nOutlier thresholds & counts per filter:")
print(thr.round(2))
thr.to_csv(ROOT / "data/outlier_thresholds.csv")

# 把异常标签回填到 daily
daily = daily.merge(thr[["lo_iqr", "hi_iqr"]].reset_index(), on="i")
daily["is_outlier"] = ((daily["y"] < daily["lo_iqr"]) | (daily["y"] > daily["hi_iqr"]))

# ---------- 作图（每台一幅） ----------
# 先载入维护事件用于竖线
mnt = pd.read_excel(PROJECT_ROOT / "题目" / "附件2.xlsx",
                    sheet_name=0, engine="openpyxl")
mnt.columns = [c.strip() for c in mnt.columns]
mnt = mnt.rename(columns={"编号": "id_str", "日期": "d", "维护类型": "q"})
mnt["i"] = mnt["id_str"].str.replace("A", "", regex=False).astype(int)
mnt["d"] = pd.to_datetime(mnt["d"])
mnt["q"] = mnt["q"].map({"小维护": "s", "中维护": "m", "大维护": "l"})
print(f"\nMaintenance records: {len(mnt)};  by type: "
      f"{mnt['q'].value_counts().to_dict()}")
print(f"Per-filter large maintenance count:\n"
      f"{mnt[mnt['q']=='l'].groupby('i').size().reindex(range(1,11), fill_value=0).to_string()}")

fig_dir = ROOT / "figures"
fig_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
for ax, i in zip(axes.ravel(), range(1, 11)):
    sub = daily[daily["i"] == i]
    ax.plot(sub["d"], sub["y"], ".", ms=2, c="steelblue", alpha=0.6)
    out = sub[sub["is_outlier"].fillna(False)]
    ax.plot(out["d"], out["y"], "x", c="red", ms=4, label=f"outlier ({len(out)})")
    # 维护竖线
    ms = mnt[mnt["i"] == i]
    for _, r in ms.iterrows():
        c = {"s": "grey", "m": "orange", "l": "darkgreen"}[r["q"]]
        ax.axvline(r["d"], c=c, alpha=0.35, lw=0.8)
    ax.axhline(37, c="red", ls="--", lw=0.5)
    ax.set_title(f"A{i} (valid={sub['y'].notna().sum()}d)", fontsize=9)
    ax.set_ylim(20, 200)
    ax.legend(loc="upper right", fontsize=7)
fig.suptitle("Daily permeability y_{i,d} (blue dots), outliers (red x), "
             "maintenance events (orange=M, green=L)", fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig1_daily_series_outliers.png", dpi=130)
plt.close(fig)
print(f"Figure saved: {fig_dir/'fig1_daily_series_outliers.png'}")

daily.to_csv(ROOT / "data/daily_long.csv", index=False)
print("Daily table updated with outlier flags.")
