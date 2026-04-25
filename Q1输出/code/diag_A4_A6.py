"""
诊断 A4 / A6 上涨现象 —— 是数据处理错误还是真实数据如此？
画出小时级原始数据 + 维护事件 + OLS 拟合线，让事实自己说话。
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")

hourly = pd.read_csv(ROOT / "Q1输出/data/hourly_long.csv", parse_dates=["h"])
hourly = hourly.rename(columns={"h":"t","p":"y"})
mnt = pd.read_csv(ROOT / "Q1输出/data/maintenance_events.csv", parse_dates=["d"])
daily = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, i in zip(axes, [4, 6]):
    sub = hourly[hourly.i == i]
    sub_d = daily[(daily.i == i) & daily.y.notna()]
    mn = mnt[mnt.i == i]

    # 小时级散点
    ax.plot(sub.t, sub.y, ".", ms=1.0, c="lightsteelblue", alpha=0.4, label="hourly")
    # 日级中位数
    ax.plot(sub_d.d, sub_d.y, "o", ms=3, c="steelblue", label="daily median")

    # 维护事件竖线
    for _, r in mn.iterrows():
        c = "darkorange" if r.q == "m" else "red"
        lw = 0.7 if r.q == "m" else 1.5
        ax.axvline(r.d, color=c, lw=lw, alpha=0.6, ls="--" if r.q=="m" else "-")

    # 月均值线（红色加粗）
    sub_m = sub.copy()
    sub_m["ym"] = sub_m.t.dt.to_period("M").dt.start_time + pd.Timedelta(days=15)
    monthly = sub_m.groupby("ym")["y"].mean().reset_index()
    ax.plot(monthly.ym, monthly.y, "-", lw=2, c="darkred", label="monthly mean", zorder=10)

    # OLS 简单趋势线
    sub_d2 = sub_d.copy()
    sub_d2["t_days"] = (sub_d2.d - sub_d2.d.min()).dt.days
    slope, intercept = np.polyfit(sub_d2["t_days"], sub_d2["y"], 1)
    ax.plot(sub_d2.d, slope*sub_d2["t_days"] + intercept, "--",
            c="black", lw=1.5, label=f"OLS slope = {slope*365.25:+.1f}/yr")

    ax.set_title(f"A{i}  hourly raw data + maintenance events  "
                 f"(orange=medium dashed, red=large solid)", fontsize=11)
    ax.set_ylabel("permeability y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(20, 145)

axes[-1].set_xlabel("date")
fig.suptitle("Why A4 / A6 show positive OLS slope?  — raw data tells the story",
             fontsize=12, y=0.998)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(ROOT / "Q1输出/figures/diag_A4_A6_raw.png", dpi=130)
plt.close(fig)
print("Saved Q1输出/figures/diag_A4_A6_raw.png")

# 数值证据汇总
print("\n=== A4 起止水平（季度均值）===")
for i in [4, 6]:
    sub = hourly[hourly.i == i].copy()
    sub["q"] = sub.t.dt.to_period("Q")
    qtr = sub.groupby("q")["y"].agg(["mean","median","count"]).round(2)
    print(f"\nA{i}:")
    print(qtr.to_string())

print("\n=== A4, A6 在头 3 月 vs 末 3 月 的均值对比 ===")
for i in [4, 6]:
    sub = hourly[hourly.i == i].sort_values("t")
    cutoff_start = sub.t.min() + pd.Timedelta(days=90)
    cutoff_end = sub.t.max() - pd.Timedelta(days=90)
    head = sub[sub.t <= cutoff_start]["y"].mean()
    tail = sub[sub.t >= cutoff_end]["y"].mean()
    print(f"  A{i}: 头 3 月 y 均值 = {head:.2f}, 末 3 月 y 均值 = {tail:.2f}, 差 = {tail-head:+.2f}")
