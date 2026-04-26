"""
Q2.7  A4 / A8 大维护策略稳健性核查
  问题: Q2 step1 对 0 次大维护历史的 A4, A8 设 T_L = inf
        (而对 1 次大维护的 A5 用 fleet 中位 262.5 fallback)
  目的: 证明 T_L 选择对寿命结论的影响有限，T_L = inf 的决策稳健

  输出: tables/A4_A8_robust_TL.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# -------- 载入 Q2 模型与历史数据（本脚本自包含，不再从 Q3 导入） --------
coef = pd.read_csv(ROOT / "Q2输出/tables/Q2_winner_coeffs.csv")
beta_dict = dict(zip(coef["var"], coef["beta"]))

mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])
hist = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
anchor = hist["d"].min()

# -------- 未来仿真时间轴 --------
start_future = pd.Timestamp("2026-01-20")
H = 4383  # 12 年 (与 Q3 地平线一致)
future_days = pd.date_range(start_future, periods=H, freq="D")
n_future = len(future_days)
T_year = 365.25
future_t = (future_days - anchor).days.values.astype(float)
future_sin1 = np.sin(2 * np.pi * future_t / T_year)
future_cos1 = np.cos(2 * np.pi * future_t / T_year)
d_arr = future_days.values


def H_window_from_dates(dates, w=7):
    """给定维护日期列表，返回未来区间内维护后 w 天窗口指示变量。"""
    flag = np.zeros(n_future, dtype=np.int8)
    for tau in dates:
        tau64 = np.datetime64(tau)
        tau_end = np.datetime64(tau + pd.Timedelta(days=w))
        mask = (d_arr > tau64) & (d_arr <= tau_end)
        flag[mask] = 1
    return flag


def A_from_dates(dates):
    """给定维护日期列表，返回未来每天距最近一次维护的天数。"""
    out = np.full(n_future, np.nan)
    dates_sorted = sorted(dates)
    if not dates_sorted:
        return out
    dates_np = np.array([np.datetime64(x) for x in dates_sorted])
    idx = np.searchsorted(dates_np, d_arr, side="right") - 1
    valid = idx >= 0
    if valid.any():
        last = dates_np[idx[valid]]
        out[valid] = (d_arr[valid] - last).astype("timedelta64[D]").astype(float)
    return out


def simulate_y(i, T_M, T_L):
    """给定设备 i 和维护周期 (T_M, T_L)，模拟未来透水率序列。"""
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
        base_l = last_l if last_l is not None else last_m
        nxt = base_l + pd.Timedelta(days=int(round(T_L)))
        while nxt <= future_days[-1]:
            future_l.append(nxt)
            nxt += pd.Timedelta(days=int(round(T_L)))

    all_m = past_m + future_m
    all_l = past_l + future_l

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
    """用历史观测 + 未来模拟的 365 日滚动均值计算寿命天数。"""
    h_sub = hist[hist["i"] == i][["d", "y"]].dropna().sort_values("d")
    h_days = h_sub["d"].values
    h_y = h_sub["y"].values

    all_dates = np.concatenate([
        h_days.astype("datetime64[D]"),
        d_arr.astype("datetime64[D]"),
    ])
    all_y = np.concatenate([h_y, y_sim])
    order = np.argsort(all_dates)
    all_dates = all_dates[order]
    all_y = all_y[order]

    _, uniq_idx = np.unique(all_dates, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    all_dates = all_dates[uniq_idx]
    all_y = all_y[uniq_idx]

    s = pd.Series(all_y, index=pd.to_datetime(all_dates)).asfreq("D")
    ravg = s.rolling(window=365, min_periods=180).mean()

    ravg_fut = ravg[ravg.index >= start_future]
    below = ravg_fut[ravg_fut < 37]
    if len(below) == 0:
        return np.inf, ravg_fut.iloc[-1] if len(ravg_fut) else np.nan
    L_date = below.index[0]
    L_days = (L_date - start_future).days
    return L_days, below.iloc[0]

rule = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")

T_L_options = [
    ("inf",       np.inf,  "Q2 现行 (不做大维护)"),
    ("262.5",     262.5,   "fleet 中位 fallback (与 A5 一致)"),
    ("365",       365.0,   "每年 1 次大维护"),
    ("180",       180.0,   "每年 2 次大维护"),
]

rows = []
for i in [4, 8]:
    r = rule[rule.i == i].iloc[0]
    T_M = float(r["T_M_use"])
    print(f"\n=== A{i} (T_M = {T_M:.0f} 天) ===")
    for label, T_L, note in T_L_options:
        y_sim, _, _ = simulate_y(i, T_M, T_L)
        L_days, _ = compute_life_days(i, y_sim)
        L_years = L_days / 365.25 if np.isfinite(L_days) else np.inf
        L_str = "inf" if np.isinf(L_years) else f"{L_years:.2f}"
        print(f"  T_L = {label:>6s} → 寿命 = {L_str:>6s} 年   ({note})")
        rows.append(dict(
            i=i,
            T_L_label=label,
            T_L_value=T_L if np.isfinite(T_L) else 99999,
            L_years=L_years if np.isfinite(L_years) else np.inf,
            L_years_display=L_str,
            note=note,
        ))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "Q2输出/tables/A4_A8_robust_TL.csv", index=False)
print("\nSaved: Q2输出/tables/A4_A8_robust_TL.csv")

# 关键结论
print("\n=== 关键结论 ===")
for i in [4, 8]:
    sub = df[df.i == i]
    L_inf = sub[sub.T_L_label == "inf"]["L_years"].iloc[0]
    L_alt = sub[sub.T_L_label == "262.5"]["L_years"].iloc[0]
    if np.isinf(L_inf) and np.isinf(L_alt):
        delta = "无差"
    else:
        delta = f"+{L_alt - L_inf:.2f} 年"
    print(f"  A{i}: T_L=inf vs T_L=262.5 → 寿命变化 {delta}")
