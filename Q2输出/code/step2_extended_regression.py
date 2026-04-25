"""
Q2.2 扩展回归（加入 γ_m·C_m + γ_l·C_l）+ 训练/验证切分
  训练：2024-04-03 至 2025-09-30
  验证：2025-10-01 至 2026-01-19（保留 2026-01-20 起的缺测段）
  比较: base (无 γ) vs extended (含 γ)  →  验证集 RMSE
输出:
  data/Q2_design_with_C.csv
  tables/Q2_regression_compare.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
df = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])

# 为每 (i, d) 添加 C_m, C_l (累计维护次数)
def add_cum(df, mnt):
    out = []
    for i, sub in df.groupby("i"):
        sub = sub.sort_values("d").reset_index(drop=True).copy()
        ev = mnt[mnt["i"] == i].sort_values("d")
        m_dates = ev[ev["q"] == "m"]["d"].values
        l_dates = ev[ev["q"] == "l"]["d"].values
        d_arr = sub["d"].values
        sub["C_m"] = np.array([(m_dates <= d).sum() for d in d_arr])
        sub["C_l"] = np.array([(l_dates <= d).sum() for d in d_arr])
        out.append(sub)
    return pd.concat(out, ignore_index=True)

df = add_cum(df, mnt)
print(f"Added C_m (range {df['C_m'].min()}..{df['C_m'].max()}), "
      f"C_l (range {df['C_l'].min()}..{df['C_l'].max()})")

df.to_csv(ROOT / "Q2输出/data/Q2_design_with_C.csv", index=False)

# 只保留 y 非空
df = df.dropna(subset=["y"]).copy()
df["A_m_f"] = df["A_m"].fillna(0)
df["A_l_f"] = df["A_l"].fillna(0)
df["has_m"] = df["A_m"].notna().astype(int)
df["has_l"] = df["A_l"].notna().astype(int)

# 训练/验证切分
train_end = pd.Timestamp("2025-09-30")
val_start = pd.Timestamp("2025-10-01")
val_end   = pd.Timestamp("2026-01-19")

df_tr = df[df["d"] <= train_end].copy()
df_va = df[(df["d"] >= val_start) & (df["d"] <= val_end)].copy()
print(f"Train: {len(df_tr)} rows  ({df_tr['d'].min().date()} .. {df_tr['d'].max().date()})")
print(f"Valid: {len(df_va)} rows  ({df_va['d'].min().date()} .. {df_va['d'].max().date()})")

filters = sorted(df["i"].unique())

def design(df, include_gamma=False):
    parts = [np.ones(len(df))]; names = ["const"]
    # 过滤器 FE
    for i in filters[1:]:
        parts.append((df["i"] == i).astype(float).values); names.append(f"I(i={i})")
    # β_i·t
    for i in filters:
        parts.append(((df["i"] == i).astype(float) * df["t"]).values); names.append(f"t_i{i}")
    # 季节
    parts.append(df["sin1"].values); names.append("sin1")
    parts.append(df["cos1"].values); names.append("cos1")
    # 维护短期 & 漂移
    for col in ["H_m7", "H_l7", "A_m_f", "A_l_f", "has_m", "has_l"]:
        parts.append(df[col].values.astype(float)); names.append(col)
    if include_gamma:
        parts.append(df["C_m"].values.astype(float)); names.append("C_m")
        parts.append(df["C_l"].values.astype(float)); names.append("C_l")
    return np.column_stack(parts), names

def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid**2).sum() / (n - k)
    try:
        var = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var))
    except Exception:
        se = np.full(k, np.nan)
    tstat = beta / np.where(se == 0, np.nan, se)
    r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
    rmse = np.sqrt((resid**2).mean())
    return beta, se, tstat, r2, rmse

def eval_rmse(beta, X, y):
    r = y - X @ beta
    return np.sqrt((r**2).mean())

results = {}
for tag, inc in [("base", False), ("extended", True)]:
    Xtr, names = design(df_tr, include_gamma=inc)
    ytr = df_tr["y"].values
    beta, se, t, r2_tr, rmse_tr = ols_fit(Xtr, ytr)
    # 验证
    Xva, _ = design(df_va, include_gamma=inc)
    yva = df_va["y"].values
    rmse_va = eval_rmse(beta, Xva, yva)
    results[tag] = dict(beta=beta, names=names, r2_tr=r2_tr, rmse_tr=rmse_tr, rmse_va=rmse_va)
    print(f"\n{tag.upper()}:")
    print(f"  train R² = {r2_tr:.4f}, train RMSE = {rmse_tr:.3f}")
    print(f"  valid RMSE = {rmse_va:.3f}")
    if inc:
        j_m = names.index("C_m"); j_l = names.index("C_l")
        print(f"  γ_m = {beta[j_m]:+.3f} (SE={se[j_m]:.3f}, t={t[j_m]:+.2f})")
        print(f"  γ_l = {beta[j_l]:+.3f} (SE={se[j_l]:.3f}, t={t[j_l]:+.2f})")

# 汇总输出
rows = []
for tag in ["base", "extended"]:
    r = results[tag]
    rows.append(dict(model=tag,
                     train_R2=r["r2_tr"], train_RMSE=r["rmse_tr"],
                     valid_RMSE=r["rmse_va"]))
cmp = pd.DataFrame(rows).round(4)
print("\n对比：")
print(cmp)
cmp.to_csv(ROOT / "Q2输出/tables/Q2_regression_compare.csv", index=False)

# 保存 extended 的完整系数
ex = results["extended"]
tab = pd.DataFrame({"var": ex["names"], "beta": ex["beta"]}).round(5)
tab.to_csv(ROOT / "Q2输出/tables/Q2_extended_coeffs.csv", index=False)
print(f"\nSaved: Q2_regression_compare.csv,  Q2_extended_coeffs.csv")
