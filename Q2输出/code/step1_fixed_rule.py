"""
Q2.1 从附件2 推导每台过滤器的"当前固定维护规律"
  - T_M_i: 历史中维护平均间隔（天）
  - T_L_i: 历史大维护平均间隔；若 <2 次大维护，标记为 inf
  - 也计算中位数作为鲁棒估计
输出:
  tables/fixed_maintenance_rule.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])
print(f"Loaded {len(mnt)} maintenance events")

rows = []
for i in sorted(mnt["i"].unique()):
    sub = mnt[mnt["i"] == i].sort_values("d")
    m_dates = sub[sub["q"] == "m"]["d"].tolist()
    l_dates = sub[sub["q"] == "l"]["d"].tolist()

    # 中维护间隔
    if len(m_dates) >= 2:
        m_int = np.diff([d.toordinal() for d in m_dates])
        T_M_mean = np.mean(m_int)
        T_M_med = np.median(m_int)
    else:
        T_M_mean = T_M_med = np.nan

    # 大维护间隔
    if len(l_dates) >= 2:
        l_int = np.diff([d.toordinal() for d in l_dates])
        T_L_mean = np.mean(l_int)
        T_L_med = np.median(l_int)
    elif len(l_dates) == 1:
        # 只观测到 1 次，无法推间隔
        T_L_mean = T_L_med = np.nan
    else:
        # 0 次：未来也不做大维护
        T_L_mean = T_L_med = np.inf

    rows.append(dict(i=i, n_m=len(m_dates), n_l=len(l_dates),
                     T_M_mean=T_M_mean, T_M_med=T_M_med,
                     T_L_mean=T_L_mean, T_L_med=T_L_med,
                     last_m=m_dates[-1] if m_dates else pd.NaT,
                     last_l=l_dates[-1] if l_dates else pd.NaT))

rule = pd.DataFrame(rows)
print("\nPer-filter empirical maintenance cadence:")
print(rule.to_string(index=False))

# 对只有 1 次大维护的台（仅 A5），用其余台的中位 T_L 填充
valid_TL = rule.loc[rule["T_L_mean"].notna() & np.isfinite(rule["T_L_mean"]),
                    "T_L_mean"]
T_L_fallback = np.median(valid_TL) if len(valid_TL) > 0 else 365.0
print(f"\nFallback T_L (median of valid): {T_L_fallback:.1f} days")

rule["T_M_use"] = rule["T_M_med"]
rule["T_L_use"] = rule["T_L_med"].copy()
mask_nan = rule["T_L_use"].isna()
rule.loc[mask_nan, "T_L_use"] = T_L_fallback
# 对 0 大维护的保持 inf
rule.loc[np.isinf(rule["T_L_mean"]), "T_L_use"] = np.inf

print("\nFinal 固定维护规律 (T_M_use, T_L_use)：")
print(rule[["i", "n_m", "n_l", "T_M_use", "T_L_use"]].to_string(index=False))

rule.to_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv", index=False)
print(f"\nSaved: {ROOT/'Q2输出/tables/fixed_maintenance_rule.csv'}")
