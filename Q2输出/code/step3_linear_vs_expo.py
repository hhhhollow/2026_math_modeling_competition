"""
Q2.3 分段线性衰减 vs 分段指数衰减比较
  线性: y_{i,d} = X·β + ε
  指数: log(y_{i,d}) = X·β' + ε'   →  y = exp(X·β')
  同一设计矩阵，比较训练 RMSE、验证 RMSE、残差结构
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
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

def design(df):
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

Xtr, names = design(df_tr)
Xva, _     = design(df_va)
ytr = df_tr["y"].values
yva = df_va["y"].values

def fit_eval(y_fit_tr, y_fit_va, transform="none"):
    beta, *_ = np.linalg.lstsq(Xtr, y_fit_tr, rcond=None)
    yhat_tr_raw = Xtr @ beta
    yhat_va_raw = Xva @ beta
    if transform == "log":
        yhat_tr = np.exp(yhat_tr_raw)
        yhat_va = np.exp(yhat_va_raw)
    else:
        yhat_tr = yhat_tr_raw
        yhat_va = yhat_va_raw
    r2_tr = 1 - ((ytr - yhat_tr)**2).sum() / ((ytr - ytr.mean())**2).sum()
    rmse_tr = np.sqrt(((ytr - yhat_tr)**2).mean())
    rmse_va = np.sqrt(((yva - yhat_va)**2).mean())
    mae_va = np.abs(yva - yhat_va).mean()
    return dict(beta=beta, r2_tr=r2_tr, rmse_tr=rmse_tr,
                rmse_va=rmse_va, mae_va=mae_va,
                yhat_tr=yhat_tr, yhat_va=yhat_va)

# 线性: 直接拟合 y
print("=== 线性模型 ===")
lin = fit_eval(ytr, yva, transform="none")
print(f"train R² = {lin['r2_tr']:.4f}, train RMSE = {lin['rmse_tr']:.3f}, "
      f"valid RMSE = {lin['rmse_va']:.3f}, valid MAE = {lin['mae_va']:.3f}")

# 指数: 拟合 log(y)
print("\n=== 指数模型 ===")
exp_model = fit_eval(np.log(ytr), np.log(yva), transform="log")
print(f"train R² = {exp_model['r2_tr']:.4f}, train RMSE = {exp_model['rmse_tr']:.3f}, "
      f"valid RMSE = {exp_model['rmse_va']:.3f}, valid MAE = {exp_model['mae_va']:.3f}")

# 分台比较：每台的 train RMSE / valid RMSE
df_tr["yhat_lin"] = lin["yhat_tr"]
df_tr["yhat_exp"] = exp_model["yhat_tr"]
df_va["yhat_lin"] = lin["yhat_va"]
df_va["yhat_exp"] = exp_model["yhat_va"]

rows = []
for i in filters:
    tr_i = df_tr[df_tr["i"] == i]
    va_i = df_va[df_va["i"] == i]
    def rmse(s1, s2): return np.sqrt(((s1 - s2)**2).mean())
    rows.append(dict(i=i,
                     n_tr=len(tr_i), n_va=len(va_i),
                     lin_tr=rmse(tr_i["y"], tr_i["yhat_lin"]),
                     exp_tr=rmse(tr_i["y"], tr_i["yhat_exp"]),
                     lin_va=rmse(va_i["y"], va_i["yhat_lin"]),
                     exp_va=rmse(va_i["y"], va_i["yhat_exp"])))
per_filter = pd.DataFrame(rows).round(3)
print("\n分台比较 (RMSE):")
print(per_filter.to_string(index=False))
per_filter.to_csv(ROOT / "Q2输出/tables/Q2_lin_vs_exp_per_filter.csv", index=False)

# 总体对比
overall = pd.DataFrame([
    dict(model="linear",      train_RMSE=lin["rmse_tr"],       valid_RMSE=lin["rmse_va"],       valid_MAE=lin["mae_va"]),
    dict(model="exponential", train_RMSE=exp_model["rmse_tr"], valid_RMSE=exp_model["rmse_va"], valid_MAE=exp_model["mae_va"]),
]).round(4)
print("\n总体：")
print(overall)
overall.to_csv(ROOT / "Q2输出/tables/Q2_lin_vs_exp_overall.csv", index=False)

# 保存最终选中的模型系数
winner = "linear" if lin["rmse_va"] <= exp_model["rmse_va"] else "exponential"
print(f"\n→ 择优: {winner}  (valid RMSE = "
      f"{lin['rmse_va'] if winner=='linear' else exp_model['rmse_va']:.3f})")
final_beta = lin["beta"] if winner == "linear" else exp_model["beta"]
pd.DataFrame({"var": names, "beta": final_beta}).to_csv(
    ROOT / "Q2输出/tables/Q2_winner_coeffs.csv", index=False)
with open(ROOT / "Q2输出/tables/Q2_winner.txt", "w") as f:
    f.write(winner)
