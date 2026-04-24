"""
Q1.4 读取附件2 并构造维护协变量
输出:
  data/maintenance_events.csv  (i, d, q)
  data/daily_with_vars.csv     (i, d, y, n, is_outlier, u_m, u_l,
                                H_m7, H_l7, A_m, A_l, sin, cos, t)
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"

# ---------- 维护事件 ----------
mnt = pd.read_excel(PROJECT_ROOT / "A题" / "附件2.xlsx",
                    sheet_name=0, engine="openpyxl")
mnt.columns = [c.strip() for c in mnt.columns]
mnt = mnt.rename(columns={"编号": "id_str", "日期": "d", "维护类型": "q"})
mnt["i"] = mnt["id_str"].str.replace("A", "", regex=False).astype(int)
mnt["d"] = pd.to_datetime(mnt["d"]).dt.normalize()
mnt["q"] = mnt["q"].map({"小维护": "s", "中维护": "m", "大维护": "l"})
mnt = mnt[["i", "d", "q"]].sort_values(["i", "d"]).reset_index(drop=True)
print(f"Maintenance events: {len(mnt)};  distribution:\n{mnt['q'].value_counts()}")
mnt.to_csv(ROOT / "data/maintenance_events.csv", index=False)

# ---------- 日表 ----------
daily = pd.read_csv(ROOT / "data/daily_long.csv", parse_dates=["d"])
daily = daily.sort_values(["i", "d"]).reset_index(drop=True)

# ---------- 构造 u^q, H^{q,7}, A^q ----------
W = 7

def build_for_filter(i, sub):
    sub = sub.sort_values("d").reset_index(drop=True).copy()
    events = mnt[mnt["i"] == i].sort_values("d")
    # u^q: 当天是否维护
    sub["u_m"] = sub["d"].isin(events[events["q"] == "m"]["d"]).astype(int)
    sub["u_l"] = sub["d"].isin(events[events["q"] == "l"]["d"]).astype(int)

    # H^{q,7}: 最近一次 q 类维护后 1..W 天窗口（不含当天 τ）
    def window_flag(q):
        flag = np.zeros(len(sub), dtype=int)
        days = events[events["q"] == q]["d"].tolist()
        d_arr = sub["d"].values
        for tau in days:
            mask = (d_arr > np.datetime64(tau)) & (d_arr <= np.datetime64(tau + pd.Timedelta(days=W)))
            flag[mask] = 1
        return flag

    sub["H_m7"] = window_flag("m")
    sub["H_l7"] = window_flag("l")

    # A^q: 距上次 q 类维护的天数（维护当天 = 0；若尚未有过维护，则为 NaN）
    def days_since(q):
        last = pd.NaT
        out = np.full(len(sub), np.nan)
        ev_days = set(events[events["q"] == q]["d"].tolist())
        for j, d in enumerate(sub["d"]):
            if d in ev_days:
                last = d
            if pd.notna(last):
                out[j] = (d - last).days
        return out

    sub["A_m"] = days_since("m")
    sub["A_l"] = days_since("l")
    return sub

out = []
for i, sub in daily.groupby("i"):
    out.append(build_for_filter(i, sub))
daily2 = pd.concat(out, ignore_index=True)

# ---------- 时间/季节协变量 ----------
anchor = daily2["d"].min()
daily2["t"] = (daily2["d"] - anchor).dt.days.astype(float)   # 天数
T = 365.25
daily2["sin1"] = np.sin(2*np.pi*daily2["t"]/T)
daily2["cos1"] = np.cos(2*np.pi*daily2["t"]/T)

# ---------- 季节标签（便于分桶） ----------
m = daily2["d"].dt.month
daily2["season"] = np.select(
    [m.isin([3,4,5]), m.isin([6,7,8]), m.isin([9,10,11])],
    ["春", "夏", "秋"], default="冬"
)

print(f"\nDaily enriched table: {len(daily2)} rows")
print("Maintenance flag counts:")
print(daily2[["u_m", "u_l", "H_m7", "H_l7"]].sum())
print(f"A_m range: {daily2['A_m'].min()} .. {daily2['A_m'].max()};  "
      f"A_l range: {daily2['A_l'].min()} .. {daily2['A_l'].max()}")

daily2.to_csv(ROOT / "data/daily_with_vars.csv", index=False)
print(f"\nSaved: {ROOT/'data/daily_with_vars.csv'}")
