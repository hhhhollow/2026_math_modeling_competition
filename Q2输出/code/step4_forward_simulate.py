"""
Q2.4 按固定维护规律向前外推 10 台透水率轨迹
  起点: 2026-01-20 (紧接观测末尾)
  外推期: 10 年 (3650 天)
  维护规则: T_M_i, T_L_i 来自 Q2.1
输出:
  data/future_trajectories.csv (i, d, y_sim)
  data/future_with_events.csv  (含未来维护事件)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)

# 载入模型系数
coef = pd.read_csv(ROOT / "Q2输出/tables/Q2_winner_coeffs.csv")
beta_dict = dict(zip(coef["var"], coef["beta"]))
print(f"Loaded {len(beta_dict)} coefficients")

# 载入固定维护规律
rule = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")
rule["T_M_use"] = rule["T_M_use"].astype(float)
rule["T_L_use"] = rule["T_L_use"].astype(float)
print(rule[["i","T_M_use","T_L_use"]])

# 历史维护记录（用于 A_m, A_l, C_m, C_l 初始化）
mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])

# 历史日表（用于获取 t 的锚点）
hist = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
anchor = hist["d"].min()  # 2024-04-03

# 外推配置
start_future = pd.Timestamp("2026-01-20")
H = 3650  # 10 年
future_days = pd.date_range(start_future, periods=H, freq="D")
T_year = 365.25

def simulate_filter(i, T_M, T_L, seed=0):
    """返回 DataFrame (d, y_sim, u_m, u_l, H_m7, H_l7, A_m, A_l)"""
    # 过去的维护
    past = mnt[mnt["i"] == i].sort_values("d")
    past_m = past[past["q"] == "m"]["d"].tolist()
    past_l = past[past["q"] == "l"]["d"].tolist()

    # 生成未来维护日程：从上次中维护日期 + T_M 开始
    last_m = past_m[-1] if past_m else anchor
    last_l = past_l[-1] if past_l else None

    future_m = []
    nxt = last_m + pd.Timedelta(days=int(round(T_M)))
    while nxt < future_days[-1]:
        future_m.append(nxt)
        nxt += pd.Timedelta(days=int(round(T_M)))

    future_l = []
    if np.isfinite(T_L):
        if last_l is None:
            # 0 大维护但 T_L 有限——不发生
            pass
        else:
            nxt = last_l + pd.Timedelta(days=int(round(T_L)))
            while nxt < future_days[-1]:
                future_l.append(nxt)
                nxt += pd.Timedelta(days=int(round(T_L)))

    all_m = past_m + future_m
    all_l = past_l + future_l

    # 为 future_days 计算每日协变量
    n = len(future_days)
    rec = pd.DataFrame(dict(d=future_days))
    rec["t"] = (rec["d"] - anchor).dt.days.astype(float)
    rec["sin1"] = np.sin(2*np.pi*rec["t"]/T_year)
    rec["cos1"] = np.cos(2*np.pi*rec["t"]/T_year)

    # u, H, A
    d_arr = rec["d"].values
    rec["u_m"] = [d in future_m for d in rec["d"]]
    rec["u_m"] = rec["u_m"].astype(int)
    rec["u_l"] = [d in future_l for d in rec["d"]]
    rec["u_l"] = rec["u_l"].astype(int)

    def H_window(dates, w=7):
        flag = np.zeros(n, dtype=int)
        for tau in dates:
            mask = (d_arr > np.datetime64(tau)) & \
                   (d_arr <= np.datetime64(tau + pd.Timedelta(days=w)))
            flag[mask] = 1
        return flag

    rec["H_m7"] = H_window(all_m)
    rec["H_l7"] = H_window(all_l)

    def days_since(dates):
        out = np.full(n, np.nan)
        last = None
        dates_sorted = sorted(dates)
        pt = 0
        for j, d in enumerate(rec["d"]):
            while pt < len(dates_sorted) and dates_sorted[pt] <= d:
                last = dates_sorted[pt]; pt += 1
            if last is not None:
                out[j] = (d - last).days
        return out

    rec["A_m"] = days_since(all_m)
    rec["A_l"] = days_since(all_l)
    rec["has_m"] = (~np.isnan(rec["A_m"])).astype(int)
    rec["has_l"] = (~np.isnan(rec["A_l"])).astype(int)
    rec["A_m_f"] = rec["A_m"].fillna(0)
    rec["A_l_f"] = rec["A_l"].fillna(0)

    # 计算 y_sim
    # 设计矩阵需要：const, I(i=2..10), t_ik (only t_i{i}=t, others=0), sin1, cos1, H_m7, H_l7, A_m_f, A_l_f, has_m, has_l
    y = np.full(n, beta_dict.get("const", 0.0))
    # 过滤器 FE
    if f"I(i={i})" in beta_dict:
        y += beta_dict[f"I(i={i})"]
    # β_i·t
    y += beta_dict[f"t_i{i}"] * rec["t"].values
    # 季节
    y += beta_dict["sin1"] * rec["sin1"].values
    y += beta_dict["cos1"] * rec["cos1"].values
    # 维护
    for col in ["H_m7", "H_l7", "A_m_f", "A_l_f", "has_m", "has_l"]:
        y += beta_dict[col] * rec[col].values

    rec["y_sim"] = y
    rec["i"] = i

    # 记录未来维护事件（用于画图）
    return rec[["i","d","t","y_sim","u_m","u_l","H_m7","H_l7","A_m","A_l",
                "has_m","has_l"]], future_m, future_l

all_trajs = []
future_events = []
for _, r in rule.iterrows():
    i = int(r["i"])
    T_M = r["T_M_use"]
    T_L = r["T_L_use"]
    traj, fm, fl = simulate_filter(i, T_M, T_L)
    all_trajs.append(traj)
    for d in fm: future_events.append(dict(i=i, d=d, q="m"))
    for d in fl: future_events.append(dict(i=i, d=d, q="l"))

fut = pd.concat(all_trajs, ignore_index=True)
fut.to_csv(ROOT / "Q2输出/data/future_trajectories.csv", index=False)

ev_df = pd.DataFrame(future_events).sort_values(["i","d"])
ev_df.to_csv(ROOT / "Q2输出/data/future_events.csv", index=False)

print(f"\nSimulated {len(fut)} future rows across {fut['i'].nunique()} filters")
print(f"Future maintenance events: {len(ev_df)}")
print(f"Time range: {fut['d'].min().date()} .. {fut['d'].max().date()}")

# 快速预览：每台第 1/12/60/120/365/1825 天的 y_sim
snap = fut.copy()
snap["days_from_start"] = (snap["d"] - start_future).dt.days
for k in [0, 30, 90, 180, 365, 730, 1095, 1460, 1825, 2555, 3285]:
    subset = snap[snap["days_from_start"] == k][["i","y_sim"]].set_index("i")["y_sim"].round(1)
    print(f"Day {k:4d}: ", subset.to_dict())
