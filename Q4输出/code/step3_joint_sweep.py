"""
Q4.3  双参数联合敏感性
  c_buy × c_l 联合扫描（c_m 占比小，固定为 base）
  - 网格 7×7 = 49 场景
  - 全队 EAC 总和热力图
  - 标记最坏 / 最好场景
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
grid = pd.read_csv(ROOT / "Q3输出/tables/eac_grid_all.csv")

C_BUY_BASE = 300.0
C_M_BASE = 3.0
C_L_BASE = 12.0

levels = np.array([-30, -20, -10, 0, 10, 20, 30]) / 100  # 7 levels

def total_optimal_eac(c_buy, c_m, c_l):
    g = grid.copy()
    # N_M / N_L 为寿命内总次数（与论文符号一致）
    g["EAC_new"] = (c_buy + g["N_M"]*c_m + g["N_L"]*c_l) / g["L_years"]
    total = 0.0
    for i in range(1, 11):
        sub = g[g.i == i]
        total += sub["EAC_new"].min()
    return total

# c_buy × c_l (c_m fixed at base)
records = []
for lb in levels:
    cb = C_BUY_BASE * (1 + lb)
    for ll in levels:
        cl = C_L_BASE * (1 + ll)
        eac = total_optimal_eac(cb, C_M_BASE, cl)
        records.append(dict(
            level_buy=lb, level_l=ll,
            c_buy=cb, c_l=cl,
            EAC_total=round(eac, 2),
        ))
joint_buy_l = pd.DataFrame(records)
joint_buy_l.to_csv(ROOT / "Q4输出/tables/joint_buy_l.csv", index=False)

print("=== c_buy × c_l 联合 (c_m=3 固定) — 全队 EAC 总和 ===")
piv = joint_buy_l.pivot_table(index="level_buy", columns="level_l", values="EAC_total")
piv.index = [f"c_buy {int(l*100):+d}%" for l in piv.index]
piv.columns = [f"c_l {int(l*100):+d}%" for l in piv.columns]
print(piv.round(1).to_string())

best = joint_buy_l.loc[joint_buy_l["EAC_total"].idxmin()]
worst = joint_buy_l.loc[joint_buy_l["EAC_total"].idxmax()]
print(f"\n最好场景: c_buy {int(best['level_buy']*100):+d}%, c_l {int(best['level_l']*100):+d}% "
      f"→ EAC = {best['EAC_total']:.1f}")
print(f"最坏场景: c_buy {int(worst['level_buy']*100):+d}%, c_l {int(worst['level_l']*100):+d}% "
      f"→ EAC = {worst['EAC_total']:.1f}")
print(f"极差 = {worst['EAC_total'] - best['EAC_total']:.1f} 万元/年")

# c_buy × c_m
records = []
for lb in levels:
    cb = C_BUY_BASE * (1 + lb)
    for lm in levels:
        cm = C_M_BASE * (1 + lm)
        eac = total_optimal_eac(cb, cm, C_L_BASE)
        records.append(dict(
            level_buy=lb, level_m=lm,
            c_buy=cb, c_m=cm,
            EAC_total=round(eac, 2),
        ))
joint_buy_m = pd.DataFrame(records)
joint_buy_m.to_csv(ROOT / "Q4输出/tables/joint_buy_m.csv", index=False)
print(f"\nSaved: joint_buy_l.csv, joint_buy_m.csv")

# 三参数同时扰动 ±30% 的极端
print("\n=== 三参数同向扰动 ===")
for sign in [-1, +1]:
    s = sign * 0.30
    eac = total_optimal_eac(C_BUY_BASE*(1+s), C_M_BASE*(1+s), C_L_BASE*(1+s))
    print(f"  全部 {int(s*100):+d}%: EAC 总和 = {eac:.2f} 万元/年")
print(f"  基准:        EAC 总和 = {total_optimal_eac(C_BUY_BASE, C_M_BASE, C_L_BASE):.2f} 万元/年")
