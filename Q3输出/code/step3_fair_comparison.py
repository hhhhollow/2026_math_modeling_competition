"""
Q3.3  公平对比：当前规律（20年地平线） vs 最优规律（20年地平线）
  注：Q2 原版寿命表用了 10 年地平线，对 A2/A7 等"边缘退役"台低估了真实寿命。
  本步统一到 20 年地平线，EAC 比较才有意义。
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)

cur = pd.read_csv(ROOT / "Q3输出/tables/current_rule_20y.csv")
opt = pd.read_csv(ROOT / "Q3输出/tables/optimal_rule_per_filter.csv")

rows = []
for i in range(1, 11):
    c = cur[cur["i"] == i].iloc[0]
    o = opt[opt["i"] == i].iloc[0]

    save_abs = c["EAC"] - o["EAC"]
    save_pct = save_abs / c["EAC"] * 100

    rows.append(dict(
        i=i,
        T_M_cur=c["T_M"], T_L_cur=c["T_L"],
        L_cur=c["L_years_20h"], retired_cur=c["retired_20h"],
        EAC_cur=c["EAC"],
        T_M_opt=int(o["T_M"]), T_L_opt=o["T_L_label"],
        L_opt=o["L_years"], retired_opt=bool(o["retired"]),
        EAC_opt=o["EAC"],
        save_abs=round(save_abs, 2),
        save_pct=round(save_pct, 1),
    ))

cmp = pd.DataFrame(rows)
print("=== 公平对比：两种规律都用 20 年地平线 ===")
print(cmp.to_string(index=False))

cmp.to_csv(ROOT / "Q3输出/tables/comparison_fair.csv", index=False)

total_cur = cmp["EAC_cur"].sum()
total_opt = cmp["EAC_opt"].sum()
print(f"\n10 台合计 EAC:")
print(f"  当前规律 (Q2, 20y): {total_cur:.2f} 万元/年")
print(f"  最优规律 (Q3):      {total_opt:.2f} 万元/年")
print(f"  节省:               {total_cur-total_opt:.2f} 万元/年  "
      f"({(total_cur-total_opt)/total_cur*100:.1f}%)")

# 单独统计：退役台 vs 不退役台
retire_now = cmp[cmp["retired_cur"]]
noretire_now = cmp[~cmp["retired_cur"]]
print(f"\n20 年内退役的 {len(retire_now)} 台: EAC 节省 "
      f"{(retire_now['EAC_cur'].sum()-retire_now['EAC_opt'].sum()):.1f} 万元/年")
print(f"20 年内不退役的 {len(noretire_now)} 台 (EAC 为上限估计): "
      f"节省 {(noretire_now['EAC_cur'].sum()-noretire_now['EAC_opt'].sum()):.1f} 万元/年")
