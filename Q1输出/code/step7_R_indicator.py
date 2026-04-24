"""
Q1.7 维护效果指标 R^{q,7}_{i,k} = y_post7 - y_pre7 (excluding τ day itself)
  - pre = 前 7 天 [τ-7, τ-1]
  - post= 后 7 天 [τ+1, τ+7]
  - 两边各需至少 3 个有效 y 才计算
  - 按季节（春/夏/秋/冬）与类型（m/l）分桶
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
daily = pd.read_csv(ROOT / "data/daily_long.csv", parse_dates=["d"])
mnt = pd.read_csv(ROOT / "data/maintenance_events.csv", parse_dates=["d"])

def season_of(d):
    m = d.month
    if m in (3,4,5): return "春"
    if m in (6,7,8): return "夏"
    if m in (9,10,11): return "秋"
    return "冬"

# 以 (i, d) 为键的 y 字典
y_lookup = {(r.i, r.d): r.y for r in daily.itertuples()}

rows = []
for ev in mnt.itertuples():
    i, tau, q = ev.i, ev.d, ev.q
    pre_days  = [tau - pd.Timedelta(days=j) for j in range(1, 8)]
    post_days = [tau + pd.Timedelta(days=j) for j in range(1, 8)]
    y_pre  = [y_lookup.get((i, dd), np.nan) for dd in pre_days]
    y_post = [y_lookup.get((i, dd), np.nan) for dd in post_days]
    y_pre  = np.array(y_pre, dtype=float)
    y_post = np.array(y_post, dtype=float)
    n_pre  = np.isfinite(y_pre).sum()
    n_post = np.isfinite(y_post).sum()
    if n_pre < 3 or n_post < 3:
        continue
    R = np.nanmean(y_post) - np.nanmean(y_pre)
    rows.append(dict(i=i, d=tau, q=q, season=season_of(tau),
                     y_pre=np.nanmean(y_pre), y_post=np.nanmean(y_post),
                     n_pre=n_pre, n_post=n_post, R=R))

R = pd.DataFrame(rows)
print(f"R indicator computed on {len(R)} events (of {len(mnt)} total)")
R.to_csv(ROOT / "tables/R_events.csv", index=False)

# 按类型汇总
by_q = R.groupby("q")["R"].agg(["count", "mean", "median", "std",
                                 lambda s: s.quantile(0.1),
                                 lambda s: s.quantile(0.9)]).round(2)
by_q.columns = ["n", "mean", "median", "std", "q10", "q90"]
print("\nR by maintenance type:")
print(by_q)
by_q.to_csv(ROOT / "tables/R_by_type.csv")

# 按季节×类型汇总
by_qs = R.groupby(["q", "season"])["R"].agg(["count", "mean", "median", "std"]).round(2)
print("\nR by (type × season):")
print(by_qs)
by_qs.to_csv(ROOT / "tables/R_by_type_season.csv")

# 按过滤器×类型
by_iq = R.groupby(["i", "q"])["R"].agg(["count", "mean"]).round(2)
print("\nR by (filter × type):")
print(by_iq.to_string())
by_iq.to_csv(ROOT / "tables/R_by_filter_type.csv")

# θ 阈值（大维护改善量的 10% 分位，用于 Q2 报废判据）
l_R = R[R["q"] == "l"]["R"]
if len(l_R) > 0:
    theta_10 = np.quantile(l_R, 0.10)
    print(f"\nθ (10% quantile of large maintenance improvement) = {theta_10:.2f}")
    print(f"Negative L-maintenance count: {(l_R < 0).sum()} / {len(l_R)}")
