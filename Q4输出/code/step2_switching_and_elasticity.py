"""
Q4.2  弹性分析与最优方案切换检查
  - 每台 EAC* 关于 c_buy / c_m / c_l 的弹性（围绕基准的中心差分）
  - 标记最优 (T_M*, T_L*) 在哪些场景下切换
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
sweep = pd.read_csv(ROOT / "Q4输出/tables/sensitivity_single_param.csv")

# ===== 切换检查 =====
print("=== 最优 (T_M*, T_L*) 切换 ===")
switching = []
for i in range(1, 11):
    sub = sweep[sweep.i == i]
    base = sub[(sub.param == "c_buy") & (sub.level == 0)].iloc[0]
    base_rule = (base["T_M"], base["T_L"])
    print(f"\nA{i}  基准最优: T_M={base['T_M']}, T_L={base['T_L']}, EAC={base['EAC']:.2f}")
    for p in ["c_buy", "c_m", "c_l"]:
        psub = sub[sub.param == p].sort_values("level")
        unique_rules = psub[["level","T_M","T_L"]].drop_duplicates(subset=["T_M","T_L"])
        if len(unique_rules) > 1:
            for _, r in unique_rules.iterrows():
                rule = (r["T_M"], r["T_L"])
                marker = " ← BASE" if rule == base_rule else ""
                print(f"    {p:5s} level={int(r['level']*100):+3d}%: T_M={r['T_M']}, T_L={r['T_L']}{marker}")
                switching.append(dict(
                    i=i, param=p, level=r["level"],
                    T_M=r["T_M"], T_L=r["T_L"],
                    is_base=rule==base_rule,
                ))

sw_df = pd.DataFrame(switching)
sw_df.to_csv(ROOT / "Q4输出/tables/optimal_rule_switching.csv", index=False)
print(f"\n切换记录已保存到 optimal_rule_switching.csv ({len(sw_df)} 条)")

# ===== 每台弹性 =====
print("\n=== 每台 EAC* 弹性（围绕基准）===")
elast_rows = []
for i in range(1, 11):
    sub = sweep[sweep.i == i]
    base_eac = sub[(sub.param == "c_buy") & (sub.level == 0)].iloc[0]["EAC"]
    rec = dict(i=i, EAC_base=round(base_eac, 2))
    for p in ["c_buy", "c_m", "c_l"]:
        psub = sub[sub.param == p].sort_values("level").reset_index(drop=True)
        # central diff at level=0 (idx=3)
        e_plus = psub.iloc[4]["EAC"] - psub.iloc[3]["EAC"]
        e_minus = psub.iloc[3]["EAC"] - psub.iloc[2]["EAC"]
        dEAC = (e_plus + e_minus) / 2
        eps = (dEAC / base_eac) / 0.10
        rec[f"eps_{p}"] = round(eps, 4)
    elast_rows.append(rec)
elast = pd.DataFrame(elast_rows)
print(elast.to_string(index=False))
elast.to_csv(ROOT / "Q4输出/tables/per_filter_elasticity.csv", index=False)

# ===== 每台 ±30% 极端场景 =====
print("\n=== 每台 ±30% 极端 EAC ===")
for i in range(1, 11):
    sub = sweep[sweep.i == i]
    print(f"\nA{i}:")
    for p in ["c_buy", "c_m", "c_l"]:
        psub = sub[sub.param == p].sort_values("level")
        eac_min = psub["EAC"].iloc[0]   # -30%
        eac_base = psub[psub.level==0]["EAC"].iloc[0]
        eac_max = psub["EAC"].iloc[-1]  # +30%
        print(f"  {p:5s}: -30% → {eac_min:.1f},  base → {eac_base:.1f},  +30% → {eac_max:.1f}  "
              f"(range = {eac_max - eac_min:.1f})")
