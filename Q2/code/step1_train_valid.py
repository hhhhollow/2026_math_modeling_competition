"""
Q2.1 构造训练/验证集 + 每台的"当前维护规律"
 - train: 2024-04-03 ~ 2025-09-30
 - valid: 2025-10-01 ~ 2026-01-19  (最后一个稠密观测日)
 - 末段 2026-01-20 之后弃用 (采集频率骤降)
 - 每台的 T_M 与 T_L: 维护事件间隔的中位数
 - A4/A6: 启动期异常，仅用 2024-09 之后的数据 (避免前 5 个月的启动偏差)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")
Q1 = ROOT / "Q1输出"
Q2 = ROOT / "Q2"

daily = pd.read_csv(Q1 / "data/daily_with_vars.csv", parse_dates=["d"])
mnt   = pd.read_csv(Q1 / "data/maintenance_events.csv", parse_dates=["d"])

# ---------- 时序切分 ----------
train_end = pd.Timestamp("2025-09-30")
valid_end = pd.Timestamp("2026-01-19")

daily["split"] = np.where(daily["d"] <= train_end, "train",
                  np.where(daily["d"] <= valid_end, "valid", "tail_sparse"))

# A4/A6 启动期裁剪 (只对训练/预测生效)
startup_cut = pd.Timestamp("2024-09-01")
daily["use_for_fit"] = True
for bad in [4, 6]:
    mask = (daily["i"] == bad) & (daily["d"] < startup_cut)
    daily.loc[mask, "use_for_fit"] = False
print(f"A4/A6 启动期裁剪 {(~daily['use_for_fit']).sum()} 行")

split_counts = (daily.assign(has_y=daily["y"].notna())
                     .groupby(["i", "split"])
                     .agg(days=("d", "size"), valid_days=("has_y", "sum"))
                     .reset_index())
print("\n训练/验证集规模：")
print(split_counts.pivot_table(index="i", columns="split",
                                 values="valid_days", fill_value=0))

# ---------- 每台维护规律 ----------
rule_rows = []
for i in range(1, 11):
    sub_m = mnt[(mnt["i"] == i) & (mnt["q"] == "m")].sort_values("d")
    sub_l = mnt[(mnt["i"] == i) & (mnt["q"] == "l")].sort_values("d")
    tm_intervals = sub_m["d"].diff().dt.days.dropna().values
    tl_intervals = sub_l["d"].diff().dt.days.dropna().values
    rule_rows.append(dict(
        i=i,
        n_m=len(sub_m), n_l=len(sub_l),
        T_M_median=int(np.median(tm_intervals)) if len(tm_intervals) > 0 else None,
        T_M_mean  =float(np.mean(tm_intervals))   if len(tm_intervals) > 0 else None,
        T_L_median=int(np.median(tl_intervals)) if len(tl_intervals) > 0 else None,
        T_L_mean  =float(np.mean(tl_intervals))   if len(tl_intervals) > 0 else None,
        last_m_date=sub_m["d"].iloc[-1] if len(sub_m) else pd.NaT,
        last_l_date=sub_l["d"].iloc[-1] if len(sub_l) else pd.NaT,
    ))
rule = pd.DataFrame(rule_rows)
# 对 A4/A8 无大维护：用"从未有大维护"的替代策略
# 若历史上无 l 维护，T_L 取全局所有台的 T_L 均值
global_tl = int(rule["T_L_median"].dropna().mean())
global_tm = int(rule["T_M_median"].dropna().mean())
rule["T_M_used"] = rule["T_M_median"].fillna(global_tm).astype(int)
rule["T_L_used"] = rule["T_L_median"].fillna(global_tl).astype(int)
print(f"\n全局均值: T_M={global_tm} d,  T_L={global_tl} d (用于填补 A4/A8)")
print("\n每台'当前维护规律'（代表训练期实际行为）：")
print(rule[["i","n_m","n_l","T_M_median","T_L_median","T_M_used","T_L_used"]].to_string(index=False))

daily.to_csv(Q2 / "data/daily_splits.csv", index=False)
rule.to_csv(Q2 / "data/maintenance_rule.csv", index=False)
print(f"\nSaved: {Q2/'data/daily_splits.csv'}")
print(f"Saved: {Q2/'data/maintenance_rule.csv'}")
