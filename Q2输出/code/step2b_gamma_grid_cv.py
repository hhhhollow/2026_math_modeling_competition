"""
Q2.2b 时序交叉验证选 (γ_m, γ_l)
  - 固定 (γ_m, γ_l) 网格值
  - 在训练集上最小化 Σ(y - γ_m*C_m - γ_l*C_l - Xβ)^2 对 β
  - 在验证集评估 RMSE
  - 选验证 RMSE 最小的 (γ_m, γ_l)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")
df = pd.read_csv(ROOT / "Q2输出/data/Q2_design_with_C.csv", parse_dates=["d"])
df = df.dropna(subset=["y"]).copy()
df["A_m_f"] = df["A_m"].fillna(0)
df["A_l_f"] = df["A_l"].fillna(0)
df["has_m"] = df["A_m"].notna().astype(int)
df["has_l"] = df["A_l"].notna().astype(int)

train_end = pd.Timestamp("2025-09-30")
val_start = pd.Timestamp("2025-10-01")
val_end   = pd.Timestamp("2026-01-19")
df_tr = df[df["d"] <= train_end].copy()
df_va = df[(df["d"] >= val_start) & (df["d"] <= val_end)].copy()

filters = sorted(df["i"].unique())

def design_base(df):
    parts = [np.ones(len(df))]; names = ["const"]
    for i in filters[1:]:
        parts.append((df["i"] == i).astype(float).values); names.append(f"I(i={i})")
    for i in filters:
        parts.append(((df["i"] == i).astype(float) * df["t"]).values); names.append(f"t_i{i}")
    parts.append(df["sin1"].values); names.append("sin1")
    parts.append(df["cos1"].values); names.append("cos1")
    for col in ["H_m7", "H_l7", "A_m_f", "A_l_f", "has_m", "has_l"]:
        parts.append(df[col].values.astype(float)); names.append(col)
    return np.column_stack(parts), names

Xtr, names = design_base(df_tr)
Xva, _     = design_base(df_va)
ytr        = df_tr["y"].values
yva        = df_va["y"].values
Cm_tr, Cl_tr = df_tr["C_m"].values, df_tr["C_l"].values
Cm_va, Cl_va = df_va["C_m"].values, df_va["C_l"].values

# 网格：预期 γ<0 (负损伤)，但也允许 0 和小正值做对比
grid_m = np.array([-3, -2, -1, -0.5, -0.25, 0, 0.25])
grid_l = np.array([-10, -5, -2, -1, 0, 1, 2])

rows = []
best = (None, None, np.inf)
for gm in grid_m:
    for gl in grid_l:
        # 目标 y_adj = y - γ·C
        y_adj_tr = ytr - gm * Cm_tr - gl * Cl_tr
        beta, *_ = np.linalg.lstsq(Xtr, y_adj_tr, rcond=None)
        # 验证集预测：ŷ = X·β + γ·C
        yhat_va = Xva @ beta + gm * Cm_va + gl * Cl_va
        rmse_va = np.sqrt(((yva - yhat_va)**2).mean())
        # 训练集也记
        yhat_tr = Xtr @ beta + gm * Cm_tr + gl * Cl_tr
        rmse_tr = np.sqrt(((ytr - yhat_tr)**2).mean())
        rows.append(dict(gamma_m=gm, gamma_l=gl,
                         rmse_train=rmse_tr, rmse_valid=rmse_va))
        if rmse_va < best[2]:
            best = (gm, gl, rmse_va)

tab = pd.DataFrame(rows).round(4)
print("CV grid results:")
print(tab.pivot_table(index="gamma_m", columns="gamma_l", values="rmse_valid").round(3))
print(f"\nBest: γ_m = {best[0]}, γ_l = {best[1]}, valid RMSE = {best[2]:.3f}")

tab.to_csv(ROOT / "Q2输出/tables/gamma_cv_grid.csv", index=False)

# 使用最佳 γ 重新拟合，作为最终模型
gm, gl, _ = best
y_adj_tr = ytr - gm * Cm_tr - gl * Cl_tr
beta_final, *_ = np.linalg.lstsq(Xtr, y_adj_tr, rcond=None)
yhat_tr = Xtr @ beta_final + gm * Cm_tr + gl * Cl_tr
yhat_va = Xva @ beta_final + gm * Cm_va + gl * Cl_va
r2_tr = 1 - ((ytr-yhat_tr)**2).sum() / ((ytr-ytr.mean())**2).sum()
rmse_tr = np.sqrt(((ytr-yhat_tr)**2).mean())
rmse_va = np.sqrt(((yva-yhat_va)**2).mean())
print(f"\nFinal model:  train R²={r2_tr:.4f}, train RMSE={rmse_tr:.3f}, valid RMSE={rmse_va:.3f}")

# 保存
pd.DataFrame({"var": names, "beta": beta_final}).to_csv(
    ROOT / "Q2输出/tables/Q2_final_coeffs.csv", index=False)
pd.DataFrame([dict(gamma_m=gm, gamma_l=gl,
                   train_R2=r2_tr, train_RMSE=rmse_tr, valid_RMSE=rmse_va)]).to_csv(
    ROOT / "Q2输出/tables/Q2_final_summary.csv", index=False)
print(f"\nSaved: Q2_final_coeffs.csv, Q2_final_summary.csv, gamma_cv_grid.csv")
