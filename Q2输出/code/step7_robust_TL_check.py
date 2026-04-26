"""
Q2.7  A4 / A8 大维护策略稳健性核查
  问题: Q2 step1 对 0 次大维护历史的 A4, A8 设 T_L = inf
        (而对 1 次大维护的 A5 用 fleet 中位 262.5 fallback)
  目的: 证明 T_L 选择对寿命结论的影响有限，T_L = inf 的决策稳健

  输出: tables/A4_A8_robust_TL.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Q3输出/code"))
from step1_grid_search_eac import simulate_y, compute_life_days  # type: ignore[import-not-found]

rule = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")

T_L_options = [
    ("inf",       np.inf,  "Q2 现行 (不做大维护)"),
    ("262.5",     262.5,   "fleet 中位 fallback (与 A5 一致)"),
    ("365",       365.0,   "每年 1 次大维护"),
    ("180",       180.0,   "每年 2 次大维护"),
]

rows = []
for i in [4, 8]:
    r = rule[rule.i == i].iloc[0]
    T_M = float(r["T_M_use"])
    print(f"\n=== A{i} (T_M = {T_M:.0f} 天) ===")
    for label, T_L, note in T_L_options:
        y_sim, _, _ = simulate_y(i, T_M, T_L)
        L_days, _ = compute_life_days(i, y_sim)
        L_years = L_days / 365.25 if np.isfinite(L_days) else np.inf
        L_str = "inf" if np.isinf(L_years) else f"{L_years:.2f}"
        print(f"  T_L = {label:>6s} → 寿命 = {L_str:>6s} 年   ({note})")
        rows.append(dict(
            i=i,
            T_L_label=label,
            T_L_value=T_L if np.isfinite(T_L) else 99999,
            L_years=L_years if np.isfinite(L_years) else np.inf,
            L_years_display=L_str,
            note=note,
        ))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "Q2输出/tables/A4_A8_robust_TL.csv", index=False)
print("\nSaved: Q2输出/tables/A4_A8_robust_TL.csv")

# 关键结论
print("\n=== 关键结论 ===")
for i in [4, 8]:
    sub = df[df.i == i]
    L_inf = sub[sub.T_L_label == "inf"]["L_years"].iloc[0]
    L_alt = sub[sub.T_L_label == "262.5"]["L_years"].iloc[0]
    if np.isinf(L_inf) and np.isinf(L_alt):
        delta = "无差"
    else:
        delta = f"+{L_alt - L_inf:.2f} 年"
    print(f"  A{i}: T_L=inf vs T_L=262.5 → 寿命变化 {delta}")
