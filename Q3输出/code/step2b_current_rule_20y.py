"""
Q3.3b  用 20 年地平线重新评估"当前规律"的真实寿命和 EAC
  (Q2 只跑了 10 年，对"不退役"的 5 台给出 L=inf → 截断 10y 导致 EAC 低估)
  本步把 10 台都在 20 年地平线下重新模拟，直接对比 Q3 最优方案
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 把当前 code 目录加入 sys.path 以便 import step1_grid_search_eac
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_grid_search_eac import simulate_y, compute_life_days  # 复用

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
rule = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")

C_BUY = 300.0
C_M = 3.0
C_L = 12.0
L_CAP = 12.0  # 12 年地平线 (覆盖正常退役最长 A2 11.57y, 不再摊薄异常台 cost_buy)

rows = []
for _, r in rule.iterrows():
    i = int(r["i"])
    T_M = float(r["T_M_use"])
    T_L_raw = r["T_L_use"]
    try:
        T_L = float(T_L_raw)
    except Exception:
        T_L = np.inf
    if str(T_L_raw) == "inf":
        T_L = np.inf

    y_sim, _, _ = simulate_y(i, T_M, T_L)
    L_days, _ = compute_life_days(i, y_sim)
    if np.isinf(L_days):
        L_years = L_CAP
        retired = False
    else:
        L_years = L_days / 365.25
        retired = True

    n_M = L_years * 365.25 / T_M
    n_L = L_years * 365.25 / T_L if np.isfinite(T_L) else 0.0
    cost_total = C_BUY + n_M * C_M + n_L * C_L
    EAC = cost_total / L_years

    rows.append(dict(
        i=i, T_M=T_M,
        T_L=("inf" if np.isinf(T_L) else f"{T_L:g}"),
        L_years_20h=round(L_years, 3),
        retired_20h=retired,
        n_M=round(n_M, 2),
        n_L=round(n_L, 2),
        cost_total=round(cost_total, 2),
        EAC=round(EAC, 2),
    ))

df = pd.DataFrame(rows)
print("当前规律在 20 年地平线下的重新评估:")
print(df.to_string(index=False))
df.to_csv(ROOT / "Q3输出/tables/current_rule_20y.csv", index=False)
print("\nSaved Q3输出/tables/current_rule_20y.csv")
