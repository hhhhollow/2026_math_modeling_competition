"""
Q2.2 + Q2.3 同时实现线性版 与 指数版 两个递推模型并在 valid 集上比较 RMSE

统一形式（一步向前预测）：
  模型 A (分段线性):  y_{d+1} = y_d  -  μ_i  -  ξ_s(d)  +  η_m u_m + η_l u_l + ε
  模型 B (一阶自回归/指数类): y_{d+1} = a_i + b_i · y_d  -  ξ_s(d)  +  η_m u_m + η_l u_l + ε
     (b_i ∈ (0,1) 隐含指数衰减率 k_i = -ln(b_i))

均以 y_{d+1} 为被解释变量拟合 OLS；训练集 ≤ 2025-09-30，验证集 2025-10 ~ 2026-01-19。
输出：
  tables/model_compare_linear_exp.csv
  data/linear_params.npz, data/exp_params.npz
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")
Q1 = ROOT / "Q1输出"
Q2 = ROOT / "Q2"

daily = pd.read_csv(Q2 / "data/daily_splits.csv", parse_dates=["d"])
# 只用 use_for_fit=True 且有 y 的行
daily = daily[daily["use_for_fit"]].copy()

# 构造 y_{d+1}
daily = daily.sort_values(["i", "d"]).reset_index(drop=True)
daily["y_next"] = daily.groupby("i")["y"].shift(-1)
daily["d_next"] = daily.groupby("i")["d"].shift(-1)
daily["gap"] = (daily["d_next"] - daily["d"]).dt.days  # 有效 only if gap==1

# 训练样本：gap=1、y, y_next 均非 NaN、位于 train
fit = daily[(daily["gap"] == 1) &
            daily["y"].notna() & daily["y_next"].notna() &
            (daily["split"] == "train")].copy()
val = daily[(daily["gap"] == 1) &
            daily["y"].notna() & daily["y_next"].notna() &
            (daily["split"] == "valid")].copy()
print(f"训练样本 n = {len(fit)},  验证样本 n = {len(val)}")

FILTERS = sorted(daily["i"].unique())

def build_X(df, model):
    """model='A' (线性) or 'B' (指数)"""
    parts, names = [], []
    # 每台截距 a_i
    for i in FILTERS:
        parts.append((df["i"] == i).astype(float).values)
        names.append(f"a_{i}")
    if model == "B":
        # 每台 b_i · y_d
        for i in FILTERS:
            parts.append(((df["i"] == i).astype(float) * df["y"]).values)
            names.append(f"b_{i}")
    # 季节
    parts.append(df["sin1"].values); names.append("sin1")
    parts.append(df["cos1"].values); names.append("cos1")
    # 维护
    parts.append(df["u_m"].values.astype(float)); names.append("eta_m")
    parts.append(df["u_l"].values.astype(float)); names.append("eta_l")
    X = np.column_stack(parts)
    return X, names

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return beta, se, resid

# ======== 拟合 ========
# 模型 A: 以 Δy = y_next - y 为被解释变量
fit_A = fit.copy()
fit_A["dy"] = fit_A["y_next"] - fit_A["y"]
XA, nA = build_X(fit_A, "A")
bA, seA, resA = ols(XA, fit_A["dy"].values)
# 模型 A 预测 y_next_hat = y + XA @ bA
fit_A["y_next_hat"] = fit_A["y"].values + XA @ bA
rmse_A_train = np.sqrt(((fit_A["y_next_hat"] - fit_A["y_next"])**2).mean())
print(f"\n[模型 A: 分段线性]  系数:")
mu = {i: -bA[nA.index(f"a_{i}")] for i in FILTERS}  # a_i = -μ_i
print(f"  μ_i (daily drop)  : {dict((i, round(mu[i]*365, 2)) for i in FILTERS)} /year")
for k in ["sin1", "cos1", "eta_m", "eta_l"]:
    j = nA.index(k); print(f"  {k}: {bA[j]:+.4f}  (SE {seA[j]:.4f})")
print(f"  RMSE (train) = {rmse_A_train:.3f}")

# 模型 B: 以 y_{d+1} 为被解释变量
XB, nB = build_X(fit, "B")
bB, seB, resB = ols(XB, fit["y_next"].values)
b_i = {i: bB[nB.index(f"b_{i}")] for i in FILTERS}
a_i = {i: bB[nB.index(f"a_{i}")] for i in FILTERS}
k_i = {i: -np.log(max(b_i[i], 1e-6)) for i in FILTERS}
rmse_B_train = np.sqrt(((XB @ bB - fit["y_next"])**2).mean())
print(f"\n[模型 B: AR(1)/指数]  系数:")
print(f"  b_i (自回归) : " + ", ".join(f"{i}:{b_i[i]:.4f}" for i in FILTERS))
print(f"  k_i = -ln(b_i) /day (× 365 = /year):")
print("  " + ", ".join(f"A{i}:{k_i[i]*365:.3f}" for i in FILTERS))
for k in ["sin1", "cos1", "eta_m", "eta_l"]:
    j = nB.index(k); print(f"  {k}: {bB[j]:+.4f}  (SE {seB[j]:.4f})")
print(f"  RMSE (train) = {rmse_B_train:.3f}")

# ======== 验证集 RMSE ========
# 模型 A
XAv, _ = build_X(val, "A")
yA_pred = val["y"].values + XAv @ bA
rmse_A_val = np.sqrt(((yA_pred - val["y_next"].values)**2).mean())
# 模型 B
XBv, _ = build_X(val, "B")
yB_pred = XBv @ bB
rmse_B_val = np.sqrt(((yB_pred - val["y_next"].values)**2).mean())

# 每台 RMSE
val["yA_pred"] = yA_pred
val["yB_pred"] = yB_pred
per_i = (val.groupby("i")
            .apply(lambda g: pd.Series({
                "n_valid": len(g),
                "RMSE_A": np.sqrt(((g["yA_pred"]-g["y_next"])**2).mean()),
                "RMSE_B": np.sqrt(((g["yB_pred"]-g["y_next"])**2).mean()),
                "MAE_A" : (g["yA_pred"]-g["y_next"]).abs().mean(),
                "MAE_B" : (g["yB_pred"]-g["y_next"]).abs().mean(),
            }), include_groups=False)
            .round(3))
print(f"\n============== 验证集对比 ==============")
print(f"模型 A (线性): RMSE = {rmse_A_val:.3f}")
print(f"模型 B (指数): RMSE = {rmse_B_val:.3f}")
print(f"=> 选择: {'A (线性)' if rmse_A_val < rmse_B_val else 'B (指数)'}")
print("\n各台验证 RMSE:")
print(per_i)
per_i.to_csv(Q2 / "tables/model_compare_linear_exp.csv")

# 保存参数
np.savez(Q2 / "data/linear_params.npz", beta=bA, names=np.array(nA))
np.savez(Q2 / "data/exp_params.npz",    beta=bB, names=np.array(nB))
print(f"\nSaved params to data/linear_params.npz and data/exp_params.npz")
