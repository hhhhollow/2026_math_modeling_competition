"""
Q1.6 Q1 回归模型（不含 γ_q，按既定口径）
   y_{i,d} = α_i + β_i·t + a·sin + b·cos + η_m·H_m7 + η_l·H_l7 + ρ_m·A_m + ρ_l·A_l + ε
  - 用 filter fixed effects (dummy)
  - 允许 β_i 按台不同（交互项 i×t）
  - A_m / A_l 在无历史维护时填 0（作为参考水平），同时加 has_m / has_l 指示变量
  - 双版本：普通 OLS 与 WLS (权重=n_{i,d})
输出：
  tables/reg_summary_ols.csv
  tables/reg_summary_wls.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
df = pd.read_csv(ROOT / "data/daily_with_vars.csv", parse_dates=["d"])

# 准备变量
df = df.dropna(subset=["y"]).copy()
df["has_m"] = df["A_m"].notna().astype(int)
df["has_l"] = df["A_l"].notna().astype(int)
df["A_m_f"] = df["A_m"].fillna(0)
df["A_l_f"] = df["A_l"].fillna(0)

# 构造设计矩阵 X
# 列： intercept (global), 9个过滤器dummy (A2..A10, A1为基准), 10个 β_i·t 交互,
#      sin1, cos1, H_m7, H_l7, A_m_f, A_l_f, has_m, has_l
filters = sorted(df["i"].unique())
ref = filters[0]

def build_X(df, add_filter_specific_trend=True):
    """构建论文 Eq.(1) 对应的设计矩阵 (28 列)。

    论文符号 ↔ 代码列名 对照：
        α_i  (filter intercepts)        → const + I(i=2)..I(i=10)   [共 10 列]
        β_i · t  (filter-time slopes)   → t_i1, t_i2, ..., t_i10    [共 10 列]
        γ_1, γ_2 (季节)                 → sin1, cos1                 [共 2 列]
        δ_m H_{i,d}^{m,7}              → H_m7                        [共 1 列]
        δ_l H_{i,d}^{l,7}              → H_l7                        [共 1 列]
        ρ_m \\tilde{A}_{i,d}^m          → A_m_f                       [共 1 列]
        ρ_l \\tilde{A}_{i,d}^l          → A_l_f                       [共 1 列]
        η_m 1_i^m  (启动哑变量)         → has_m                       [共 1 列]
        η_l 1_i^l                       → has_l                       [共 1 列]
    合计 28 列。
    """
    parts = [np.ones(len(df))]
    names = ["const"]
    # ---- α_i: 过滤器固定效应 (以 A1 为基准)
    for i in filters[1:]:
        parts.append((df["i"] == i).astype(float).values)
        names.append(f"I(i={i})")
    # ---- β_i · t: 过滤器特定时间斜率
    if add_filter_specific_trend:
        for i in filters:
            parts.append(((df["i"] == i).astype(float) * df["t"]).values)
            names.append(f"t_i{i}")
    else:
        parts.append(df["t"].values); names.append("t")
    # ---- γ_1 sin + γ_2 cos: 一阶谐波季节项
    parts.append(df["sin1"].values); names.append("sin1")
    parts.append(df["cos1"].values); names.append("cos1")
    # ---- 维护项：δ_m H_m7, δ_l H_l7, ρ_m A_m_f, ρ_l A_l_f, η_m has_m, η_l has_l
    for col in ["H_m7", "H_l7", "A_m_f", "A_l_f", "has_m", "has_l"]:
        parts.append(df[col].values.astype(float)); names.append(col)
    X = np.column_stack(parts)
    return X, names

def ols(X, y, w=None):
    """最小二乘 + HC0 稳健标准误（可加权）。"""
    if w is None:
        Wy = y
        WX = X
    else:
        sw = np.sqrt(w)
        Wy = y * sw
        WX = X * sw[:, None]
    beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
    # 残差
    resid = y - X @ beta
    n, k = X.shape
    # 普通方差
    if w is None:
        sigma2 = (resid**2).sum() / (n - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        var_beta = sigma2 * XtX_inv
    else:
        # WLS: var = (X'WX)^-1 X'W diag(r^2) W X (X'WX)^-1  (HC0 variant)
        W = np.diag(w)
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        meat = X.T @ W @ np.diag(resid**2) @ W @ X
        var_beta = XtWX_inv @ meat @ XtWX_inv
    se = np.sqrt(np.diag(var_beta))
    tstat = beta / np.where(se == 0, np.nan, se)
    # R^2
    ss_res = (resid**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    return beta, se, tstat, r2, resid

X, names = build_X(df, add_filter_specific_trend=True)
y = df["y"].values
w = df["n"].values.astype(float)

b_ols, se_ols, t_ols, r2_ols, res_ols = ols(X, y, w=None)
b_wls, se_wls, t_wls, r2_wls, res_wls = ols(X, y, w=w)
print(f"OLS  R² = {r2_ols:.4f},  n = {len(y)},  k = {X.shape[1]}")
print(f"WLS  R² = {r2_wls:.4f}")

tab = pd.DataFrame({
    "var": names,
    "beta_OLS": b_ols,
    "SE_OLS": se_ols,
    "t_OLS": t_ols,
    "beta_WLS": b_wls,
    "SE_WLS": se_wls,
    "t_WLS": t_wls,
}).round(4)
tab.to_csv(ROOT / "tables/reg_summary_full.csv", index=False)

# 关键系数摘要
key = tab[tab["var"].isin(["sin1", "cos1", "H_m7", "H_l7", "A_m_f", "A_l_f"])].copy()
key["ampl(sin+cos)"] = np.where(key["var"] == "sin1",
                                 np.sqrt(b_ols[names.index("sin1")]**2 +
                                         b_ols[names.index("cos1")]**2), np.nan)
print("\nKey coefficients:")
print(tab[tab["var"].isin(["const","sin1","cos1","H_m7","H_l7","A_m_f","A_l_f","has_m","has_l"])].to_string(index=False))

# 各台趋势斜率 β_i (天/天) 与 年化
trend_rows = []
for i in filters:
    j = names.index(f"t_i{i}")
    trend_rows.append(dict(i=i, beta_per_day=b_ols[j], SE=se_ols[j],
                           beta_per_year=b_ols[j]*365.25, t=t_ols[j]))
trend = pd.DataFrame(trend_rows).round(4)
print("\nFilter-specific trend β_i (OLS):")
print(trend.to_string(index=False))
trend.to_csv(ROOT / "tables/trend_per_filter.csv", index=False)

# 季节性幅度
# y = a sin(2π t/T) + b cos(2π t/T) = R sin(2π t/T + φ)，φ = arctan2(b, a)
# 峰值 t_peak = T(π/2 − φ)/(2π) = T(90 − φ°)/360 = T·phase°/360
#   其中 phase = arctan2(a, b) = 90° − φ°（与 a, b 互调位的反正切等价变换）
# 注：t_peak 是相对 anchor 的偏移天数，**不是日历 DOY**。要换算日历日期请加上 anchor。
sin_idx = names.index("sin1"); cos_idx = names.index("cos1")
amp = np.sqrt(b_ols[sin_idx]**2 + b_ols[cos_idx]**2)
phase = np.degrees(np.arctan2(b_ols[sin_idx], b_ols[cos_idx]))
t_peak = phase / 360 * 365.25  # 相对 anchor 的偏移天数
anchor_date = df["d"].min()
peak_date = anchor_date + pd.Timedelta(days=int(round(t_peak)))
print(f"\nSeasonal amplitude = {amp:.3f},  phase = {phase:.1f}°")
print(f"  Peak at t = {t_peak:.1f} d (offset from anchor {anchor_date.date()})")
print(f"  Peak calendar date ≈ {peak_date.date()} (calendar DOY ≈ {peak_date.dayofyear})")

# 保存残差
df["resid_ols"] = res_ols
df["resid_wls"] = res_wls
df[["i","d","y","n","resid_ols","resid_wls"]].to_csv(ROOT / "data/regression_residuals.csv", index=False)
print(f"\nSaved: tables/reg_summary_full.csv,  tables/trend_per_filter.csv,  data/regression_residuals.csv")
