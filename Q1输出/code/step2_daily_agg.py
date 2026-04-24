"""
Q1.2 日聚合：y_{i,d}=median{p_i(h):h in d}, 保留 n_{i,d}
      n_{i,d}<12 时 y 置为 NaN
输出: data/daily_long.csv (columns: i, d, y, n, y_raw)
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
hourly = pd.read_csv(ROOT / "data/hourly_long.csv", parse_dates=["h"])
print(f"Loaded {len(hourly)} hourly rows")

hourly["d"] = hourly["h"].dt.normalize()  # 当天 00:00

# 丢掉 p=NaN 的小时，再做 median / count
valid = hourly.dropna(subset=["p"])
daily = (valid.groupby(["i", "d"])
              .agg(y_raw=("p", "median"), n=("p", "size"))
              .reset_index())

N_MIN = 12
daily["y"] = daily["y_raw"].where(daily["n"] >= N_MIN, np.nan)

# 生成完整日历 (i × all days)
all_days = pd.date_range(daily["d"].min(), daily["d"].max(), freq="D")
idx = pd.MultiIndex.from_product([sorted(daily["i"].unique()), all_days],
                                  names=["i", "d"])
daily = (daily.set_index(["i", "d"])
              .reindex(idx)
              .reset_index())
daily["n"] = daily["n"].fillna(0).astype(int)

print(f"Daily long table: {len(daily)} rows  ({daily['i'].nunique()} filters × {len(all_days)} days)")
print(f"NaN y count: {daily['y'].isna().sum()}  "
      f"({100*daily['y'].isna().mean():.1f}% of rows)")
print(f"Rows with n<12 (masked): {((daily['n']>0)&(daily['n']<12)).sum()}")

# 每台统计
summary = (daily.groupby("i")
                .agg(days_total=("d", "size"),
                     days_valid=("y", lambda s: s.notna().sum()),
                     days_n_zero=("n", lambda s: (s==0).sum()),
                     days_n_lt12=("n", lambda s: ((s>0)&(s<12)).sum()),
                     y_min=("y", "min"),
                     y_max=("y", "max"),
                     y_mean=("y", "mean"))
                .round(2))
summary["coverage_%"] = (100 * summary["days_valid"] / summary["days_total"]).round(1)
print("\nPer-filter daily summary:")
print(summary)

daily.to_csv(ROOT / "data/daily_long.csv", index=False)
summary.to_csv(ROOT / "data/per_filter_daily_summary.csv")
print(f"\nSaved: {ROOT/'data/daily_long.csv'}")
