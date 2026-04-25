"""
Q4.1  单参数敏感性扫描
  对 c_buy, c_m, c_l 各自在 ±30% 范围扫描 (-30, -20, -10, 0, +10, +20, +30)%
  每个场景下用 Q3 已存的 L/n_M/n_L 重新计算 EAC, 找最优 (T_M*, T_L*)

加速思路：寿命 L、维护次数 n_M, n_L 只依赖 (i, T_M, T_L)，与成本无关。
所以只需读 Q3 的 eac_grid_all.csv，重算 EAC 即可，无需重跑仿真。

输出:
  tables/sensitivity_single_param.csv   每场景每台的最优结果
  tables/sensitivity_total_fleet.csv    每场景全队 EAC 总和
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")
grid = pd.read_csv(ROOT / "Q3输出/tables/eac_grid_all.csv")
print(f"Loaded {len(grid)} grid rows from Q3")

C_BUY = 300.0
C_M = 3.0
C_L = 12.0

levels = np.array([-30, -20, -10, 0, 10, 20, 30]) / 100  # 7

def best_per_filter(c_buy, c_m, c_l):
    g = grid.copy()
    g["EAC_new"] = (c_buy + g["n_M"]*c_m + g["n_L"]*c_l) / g["L_years"]
    rows = []
    for i in range(1, 11):
        sub = g[g.i == i]
        b = sub.loc[sub["EAC_new"].idxmin()]
        T_L_disp = "inf" if b["T_L_label"] == np.inf or b["T_L"] == 99999 else f"{int(b['T_L_label'])}"
        rows.append(dict(
            i=i,
            T_M=int(b["T_M"]),
            T_L=T_L_disp,
            L_years=round(b["L_years"], 2),
            retired=bool(b["retired"]),
            EAC=round(b["EAC_new"], 3),
        ))
    return pd.DataFrame(rows)

# 三个参数分别扫描
all_records = []
fleet_records = []

for param_name, c_base in [("c_buy", C_BUY), ("c_m", C_M), ("c_l", C_L)]:
    for level in levels:
        c_val = c_base * (1 + level)
        if param_name == "c_buy":
            df = best_per_filter(c_val, C_M, C_L)
        elif param_name == "c_m":
            df = best_per_filter(C_BUY, c_val, C_L)
        else:  # c_l
            df = best_per_filter(C_BUY, C_M, c_val)
        df["param"] = param_name
        df["level"] = level
        df["c_value"] = round(c_val, 3)
        all_records.append(df)
        fleet_records.append(dict(
            param=param_name,
            level=level,
            c_value=round(c_val, 3),
            EAC_total=round(df["EAC"].sum(), 2),
        ))

sweep = pd.concat(all_records, ignore_index=True)
fleet = pd.DataFrame(fleet_records)

sweep.to_csv(ROOT / "Q4输出/tables/sensitivity_single_param.csv", index=False)
fleet.to_csv(ROOT / "Q4输出/tables/sensitivity_total_fleet.csv", index=False)

print(f"\nSaved:")
print(f"  sensitivity_single_param.csv  ({len(sweep)} rows)")
print(f"  sensitivity_total_fleet.csv   ({len(fleet)} rows)")

# 摘要打印：全队 EAC 在各场景下的变化
print("\n=== 全队 EAC 总和（万元/年）===")
piv = fleet.pivot_table(index="param", columns="level", values="EAC_total")
piv.columns = [f"{int(l*100):+d}%" for l in piv.columns]
print(piv.round(2).to_string())

# 弹性 (中心差分)
def elasticity(series_levels, series_eac, base_idx):
    """围绕 base 的弹性 ε = (dEAC/EAC) / (dc/c)"""
    e_plus = series_eac.iloc[base_idx + 1] - series_eac.iloc[base_idx]
    e_minus = series_eac.iloc[base_idx] - series_eac.iloc[base_idx - 1]
    dEAC = (e_plus + e_minus) / 2
    dC_pct = 0.10  # ±10% 步长
    return (dEAC / series_eac.iloc[base_idx]) / dC_pct

base_idx = 3  # level=0
print("\n=== 全队 EAC 弹性 (围绕基准点) ===")
for p in ["c_buy", "c_m", "c_l"]:
    sub = fleet[fleet.param == p].sort_values("level").reset_index(drop=True)
    eps = elasticity(sub["level"], sub["EAC_total"], base_idx)
    print(f"  {p}: ε = {eps:.4f}  "
          f"(EAC at -30%/0/+30% = {sub['EAC_total'].iloc[0]:.1f} / "
          f"{sub['EAC_total'].iloc[base_idx]:.1f} / {sub['EAC_total'].iloc[-1]:.1f})")
