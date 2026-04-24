"""
Q3.1 + Q3.2  网格搜索：每台过滤器最小 EAC 的 (T_M, T_L)

EAC_i(T_M, T_L) = (c_buy + n_M * c_m + n_L * c_l) / L_years

其中:
  c_buy = 300 万元 (新购)
  c_m   =   3 万元 (中维护)
  c_l   =  12 万元 (大维护)
  L_years = 寿命（从 2026-01-20 起到滚动 365 日均值首次 < 37）
  n_M   = L_years · 365.25 / T_M (整个寿命内中维护次数)
  n_L   = L_years · 365.25 / T_L (若 T_L 有限，否则 n_L = 0)

策略:
  为避免长寿命过滤器无法给出 L，仿真地平线延长到 20 年 (H=7300)。
  若 20 年内仍未退役，记 L = 20（上限），EAC 为保守估计（真实 EAC 更低）。

网格:
  T_M ∈ {30, 40, 50, 60, 70, 80, 90, 100, 120} (9)
  T_L ∈ {120, 180, 240, 300, 360, 450, 540, 720, ∞} (9)
  共 81 个组合 × 10 台 = 810 次模拟。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")

# -------- 载入 Q2 模型系数 --------
coef = pd.read_csv(ROOT / "Q2输出/tables/Q2_winner_coeffs.csv")
beta_dict = dict(zip(coef["var"], coef["beta"]))
print(f"Loaded {len(beta_dict)} coefficients from Q2")

# -------- 历史数据 --------
mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])
hist = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
anchor = hist["d"].min()  # 2024-04-03

# -------- 仿真参数 --------
start_future = pd.Timestamp("2026-01-20")
H = 7300  # 20 年
future_days = pd.date_range(start_future, periods=H, freq="D")
n_future = len(future_days)
T_year = 365.25
future_t = (future_days - anchor).days.values.astype(float)
future_sin1 = np.sin(2*np.pi*future_t/T_year)
future_cos1 = np.cos(2*np.pi*future_t/T_year)
d_arr = future_days.values  # numpy datetime64

# -------- 成本常数 --------
C_BUY = 300.0
C_M = 3.0
C_L = 12.0

# -------- 网格 --------
T_M_grid = [30, 40, 50, 60, 70, 80, 90, 100, 120]
T_L_grid = [120, 180, 240, 300, 360, 450, 540, 720, np.inf]

# -------- 预合并观测 y （用于滚动 365 日均值的前置窗口） --------
hist_slim = hist[["i","d","y"]].copy().rename(columns={"y":"y_obs"})

# -------- 构建 H_window 辅助 --------
def H_window_from_dates(dates, w=7):
    """给定维护日期列表，返回 n_future 长的 0/1 数组"""
    flag = np.zeros(n_future, dtype=np.int8)
    for tau in dates:
        tau64 = np.datetime64(tau)
        tau_end = np.datetime64(tau + pd.Timedelta(days=w))
        mask = (d_arr > tau64) & (d_arr <= tau_end)
        flag[mask] = 1
    return flag

def A_from_dates(dates):
    """给定维护日期列表（含历史+未来，且 <= future_days[-1]），返回每日 days-since-last"""
    out = np.full(n_future, np.nan)
    dates_sorted = sorted(dates)
    if not dates_sorted:
        return out
    # 对每个 future day，找最后一个 <= d 的 dates
    # 利用 searchsorted
    dates_np = np.array([np.datetime64(x) for x in dates_sorted])
    idx = np.searchsorted(dates_np, d_arr, side="right") - 1
    valid = idx >= 0
    if valid.any():
        last = dates_np[idx[valid]]
        diff = (d_arr[valid] - last).astype("timedelta64[D]").astype(float)
        out[valid] = diff
    return out

def simulate_y(i, T_M, T_L):
    """给定 (i, T_M, T_L)，返回 n_future 长 y_sim 数组"""
    past = mnt[mnt["i"] == i].sort_values("d")
    past_m = past[past["q"] == "m"]["d"].tolist()
    past_l = past[past["q"] == "l"]["d"].tolist()

    last_m = past_m[-1] if past_m else anchor
    last_l = past_l[-1] if past_l else None

    future_m = []
    nxt = last_m + pd.Timedelta(days=int(round(T_M)))
    while nxt <= future_days[-1]:
        future_m.append(nxt)
        nxt += pd.Timedelta(days=int(round(T_M)))

    future_l = []
    if np.isfinite(T_L):
        base_l = last_l if last_l is not None else last_m  # 若无大维护，从上次中维护起步
        nxt = base_l + pd.Timedelta(days=int(round(T_L)))
        while nxt <= future_days[-1]:
            future_l.append(nxt)
            nxt += pd.Timedelta(days=int(round(T_L)))

    all_m = past_m + future_m
    all_l = past_l + future_l

    u_m = np.isin(d_arr, np.array([np.datetime64(x) for x in future_m])).astype(np.int8)
    u_l = np.isin(d_arr, np.array([np.datetime64(x) for x in future_l])).astype(np.int8) if future_l else np.zeros(n_future, dtype=np.int8)

    H_m7 = H_window_from_dates(all_m, w=7)
    H_l7 = H_window_from_dates(all_l, w=7) if all_l else np.zeros(n_future, dtype=np.int8)

    A_m = A_from_dates(all_m)
    A_l = A_from_dates(all_l) if all_l else np.full(n_future, np.nan)
    has_m = (~np.isnan(A_m)).astype(np.int8)
    has_l = (~np.isnan(A_l)).astype(np.int8)
    A_m_f = np.where(np.isnan(A_m), 0.0, A_m)
    A_l_f = np.where(np.isnan(A_l), 0.0, A_l)

    y = np.full(n_future, beta_dict.get("const", 0.0))
    if f"I(i={i})" in beta_dict:
        y += beta_dict[f"I(i={i})"]
    y += beta_dict[f"t_i{i}"] * future_t
    y += beta_dict["sin1"] * future_sin1
    y += beta_dict["cos1"] * future_cos1
    y += beta_dict["H_m7"] * H_m7
    y += beta_dict["H_l7"] * H_l7
    y += beta_dict["A_m_f"] * A_m_f
    y += beta_dict["A_l_f"] * A_l_f
    y += beta_dict["has_m"] * has_m
    y += beta_dict["has_l"] * has_l

    return y, len(future_m), len(future_l)

def compute_life_days(i, y_sim):
    """用历史观测 + 未来模拟拼接后的序列做滚动 365d 均值，返回寿命（天）"""
    # 拼接历史 y 和未来 y_sim
    h_sub = hist[hist["i"] == i][["d","y"]].dropna().sort_values("d")
    h_days = h_sub["d"].values
    h_y = h_sub["y"].values
    # Future: future_days + y_sim
    all_dates = np.concatenate([h_days.astype("datetime64[D]"),
                                d_arr.astype("datetime64[D]")])
    all_y = np.concatenate([h_y, y_sim])
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_y = all_y[order]
    # 去重日期（若历史最后日与未来首日重叠）
    _, uniq_idx = np.unique(all_dates, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    all_dates = all_dates[uniq_idx]
    all_y = all_y[uniq_idx]

    # 将序列转为 pandas 做滚动均值（兼容 NaN 容差）
    s = pd.Series(all_y, index=pd.to_datetime(all_dates))
    # 重采样为日频并前向填充观测区间的缺失（保留滚动自然处理 NaN）
    s = s.asfreq("D")
    ravg = s.rolling(window=365, min_periods=180).mean()

    # 只在未来段找首次 < 37
    fut_mask = ravg.index >= start_future
    ravg_fut = ravg[fut_mask]
    below = ravg_fut[ravg_fut < 37]
    if len(below) == 0:
        return np.inf, ravg_fut.iloc[-1] if len(ravg_fut) else np.nan
    L_date = below.index[0]
    L_days = (L_date - start_future).days
    return L_days, below.iloc[0]

# -------- 主循环 --------
start_time = time.time()
rows = []
for i in range(1, 11):
    best = dict(EAC=np.inf)
    for T_M in T_M_grid:
        for T_L in T_L_grid:
            y_sim, _, _ = simulate_y(i, T_M, T_L)
            L_days, rav_at_L = compute_life_days(i, y_sim)
            if np.isinf(L_days):
                L_years = 20.0  # 上限
                retired = False
            else:
                L_years = L_days / 365.25
                retired = True
            # 成本
            n_M = L_years * 365.25 / T_M
            n_L = L_years * 365.25 / T_L if np.isfinite(T_L) else 0.0
            cost_total = C_BUY + n_M * C_M + n_L * C_L
            EAC = cost_total / L_years
            row = dict(
                i=i, T_M=T_M, T_L=T_L if np.isfinite(T_L) else 99999,
                T_L_label="inf" if np.isinf(T_L) else str(int(T_L)),
                L_days=int(L_days) if np.isfinite(L_days) else 99999,
                L_years=round(L_years, 3),
                retired=retired,
                n_M=round(n_M, 2),
                n_L=round(n_L, 2),
                cost_total=round(cost_total, 2),
                EAC=round(EAC, 3),
            )
            rows.append(row)
            if EAC < best["EAC"]:
                best = row.copy()
    elapsed = time.time() - start_time
    print(f"A{i:2d}: T_M*={best['T_M']}, T_L*={best['T_L_label']}, "
          f"L={best['L_years']:.2f}y, EAC*={best['EAC']:.2f} 万元/年  "
          f"(retired={best['retired']})  [{elapsed:.1f}s]")

grid = pd.DataFrame(rows)
grid.to_csv(ROOT / "Q3输出/tables/eac_grid_all.csv", index=False)
print(f"\nSaved Q3输出/tables/eac_grid_all.csv  ({len(grid)} rows)")

# -------- 每台最优 --------
best_rows = []
for i in range(1, 11):
    sub = grid[grid["i"] == i]
    best = sub.loc[sub["EAC"].idxmin()].to_dict()
    best_rows.append(best)
best_df = pd.DataFrame(best_rows)
best_df.to_csv(ROOT / "Q3输出/tables/optimal_rule_per_filter.csv", index=False)
print(f"\nOptimal rule per filter:")
print(best_df[["i","T_M","T_L_label","L_years","retired","n_M","n_L","cost_total","EAC"]].to_string(index=False))
