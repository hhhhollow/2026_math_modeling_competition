"""
Q3.3  当前规律 vs 最优规律 对比
  - 当前规律：Q2 fixed_maintenance_rule.csv 的 T_M_use, T_L_use
  - 当前 L：Q2 life_prediction.csv 的 L_year（inf 截断到 20 年作为上限）
  - 最优规律：Q3 step1 的 optimal_rule_per_filter.csv

输出：
  tables/comparison_current_vs_optimal.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)

rule_now = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")
life_now = pd.read_csv(ROOT / "Q2输出/tables/life_prediction.csv")
opt = pd.read_csv(ROOT / "Q3输出/tables/optimal_rule_per_filter.csv")

C_BUY = 300.0
C_M = 3.0
C_L = 12.0
L_CAP = 20.0  # 与 Q3 仿真地平线一致

rows = []
for i in range(1, 11):
    rn = rule_now[rule_now["i"] == i].iloc[0]
    ln = life_now[life_now["i"] == i].iloc[0]
    op = opt[opt["i"] == i].iloc[0]

    # 当前规律
    T_M_cur = rn["T_M_use"]
    T_L_cur = rn["T_L_use"] if not (isinstance(rn["T_L_use"], str) and rn["T_L_use"] == "inf") else np.inf
    try:
        T_L_cur = float(T_L_cur)
    except Exception:
        T_L_cur = np.inf

    L_cur_y_raw = ln["L_year"]
    if isinstance(L_cur_y_raw, str) and L_cur_y_raw == "inf":
        L_cur_y = L_CAP
        retired_cur = False
    elif np.isinf(L_cur_y_raw) or pd.isna(L_cur_y_raw):
        L_cur_y = L_CAP
        retired_cur = False
    else:
        L_cur_y = float(L_cur_y_raw)
        retired_cur = True

    n_M_cur = L_cur_y * 365.25 / T_M_cur
    n_L_cur = L_cur_y * 365.25 / T_L_cur if np.isfinite(T_L_cur) else 0.0
    cost_cur = C_BUY + n_M_cur * C_M + n_L_cur * C_L
    EAC_cur = cost_cur / L_cur_y

    # 最优
    T_M_opt = op["T_M"]
    T_L_opt_lbl = op["T_L_label"]
    T_L_opt = np.inf if T_L_opt_lbl == "inf" else float(T_L_opt_lbl)
    L_opt_y = op["L_years"]
    EAC_opt = op["EAC"]
    cost_opt = op["cost_total"]
    n_M_opt = op["n_M"]
    n_L_opt = op["n_L"]
    retired_opt = bool(op["retired"])

    # 节省
    save_abs = EAC_cur - EAC_opt
    save_pct = save_abs / EAC_cur * 100

    rows.append(dict(
        i=i,
        T_M_cur=T_M_cur,
        T_L_cur=("inf" if np.isinf(T_L_cur) else f"{T_L_cur:g}"),
        L_cur_years=round(L_cur_y, 2),
        retired_cur=retired_cur,
        n_M_cur=round(n_M_cur, 1),
        n_L_cur=round(n_L_cur, 1),
        cost_cur=round(cost_cur, 1),
        EAC_cur=round(EAC_cur, 2),
        T_M_opt=int(T_M_opt),
        T_L_opt=T_L_opt_lbl,
        L_opt_years=round(L_opt_y, 2),
        retired_opt=retired_opt,
        n_M_opt=round(n_M_opt, 1),
        n_L_opt=round(n_L_opt, 1),
        cost_opt=round(cost_opt, 1),
        EAC_opt=round(EAC_opt, 2),
        save_abs=round(save_abs, 2),
        save_pct=round(save_pct, 1),
    ))

cmp = pd.DataFrame(rows)
print("当前规律 vs 最优规律:")
cols_print = ["i","T_M_cur","T_L_cur","L_cur_years","EAC_cur",
              "T_M_opt","T_L_opt","L_opt_years","EAC_opt","save_abs","save_pct"]
print(cmp[cols_print].to_string(index=False))

cmp.to_csv(ROOT / "Q3输出/tables/comparison_current_vs_optimal.csv", index=False)
print("\nSaved Q3输出/tables/comparison_current_vs_optimal.csv")

# 汇总
total_cur = cmp["EAC_cur"].sum()
total_opt = cmp["EAC_opt"].sum()
total_save = total_cur - total_opt
total_save_pct = total_save / total_cur * 100
print(f"\n10 台合计年化成本：")
print(f"  当前规律: {total_cur:.2f} 万元/年")
print(f"  最优规律: {total_opt:.2f} 万元/年")
print(f"  节省:     {total_save:.2f} 万元/年  ({total_save_pct:.1f}%)")

# A4/A6 警示
print("\n⚠ 警示：A4 / A6 在 Q2 模型下历史斜率为正 (β>0)，")
print("   其外推 y 会持续上涨，寿命预测不退役，EAC 为保守下限估计。")
print("   实际工程中应按 β<0 的其它台类比处理，或使用更短历史窗口重新估 β。")
