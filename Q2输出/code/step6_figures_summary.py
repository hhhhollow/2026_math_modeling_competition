"""
Q2.6 输出图表与总结
  fig1: 10 台过滤器的 y 轨迹（历史观测 + 未来预测）
  fig2: 10 台滚动 365 日均值曲线 + 37 阈值线
  fig3: 预测寿命条形图
  Q2_summary.md: 关键数字与解读
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)

fut = pd.read_csv(ROOT / "Q2输出/data/future_trajectories.csv", parse_dates=["d"])
hist = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
merged = pd.read_csv(ROOT / "Q2输出/data/full_with_rolling.csv", parse_dates=["d"])
life = pd.read_csv(ROOT / "Q2输出/tables/life_prediction.csv", parse_dates=["L_date"])
rule = pd.read_csv(ROOT / "Q2输出/tables/fixed_maintenance_rule.csv")

start_future = pd.Timestamp("2026-01-20")

# ---------- Fig 1: 10 台 y 轨迹 ----------
fig, axes = plt.subplots(5, 2, figsize=(16, 16), sharex=True)
for ax, i in zip(axes.ravel(), range(1, 11)):
    h = hist[hist["i"] == i]
    f = fut[fut["i"] == i]
    ax.plot(h["d"], h["y"], ".", ms=1, c="steelblue", alpha=0.5, label="observed")
    ax.plot(f["d"], f["y_sim"], "-", lw=0.6, c="crimson", alpha=0.7, label="simulated")
    ax.axhline(37, c="red", ls="--", lw=0.7)
    ax.axvline(start_future, c="k", ls=":", lw=0.6)
    r = life[life["i"] == i].iloc[0]
    if pd.notna(r["L_date"]):
        ax.axvline(r["L_date"], c="darkred", ls="-.", lw=0.9,
                   label=f"L={r['L_year']:.2f}y")
    ax.set_title(f"A{i}  T_M={rule[rule['i']==i]['T_M_use'].iloc[0]:.0f}d, "
                 f"T_L={rule[rule['i']==i]['T_L_use'].iloc[0]}d",
                 fontsize=9)
    ax.set_ylim(-50, 250)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Q2  permeability y: observed (blue) + simulated under fixed maintenance rule (red)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(ROOT / "Q2输出/figures/fig_q2_trajectories.png", dpi=110)
plt.close(fig)
print("Saved fig_q2_trajectories.png")

# ---------- Fig 2: 滚动 365 日均值 ----------
fig, axes = plt.subplots(5, 2, figsize=(16, 16), sharex=True, sharey=True)
for ax, i in zip(axes.ravel(), range(1, 11)):
    m = merged[merged["i"] == i].sort_values("d")
    ax.plot(m["d"], m["rolling_avg_365"], "-", lw=1, c="darkblue")
    ax.axhline(37, c="red", ls="--", lw=0.7, label="threshold 37")
    ax.axvline(start_future, c="k", ls=":", lw=0.6)
    r = life[life["i"] == i].iloc[0]
    if pd.notna(r["L_date"]):
        ax.axvline(r["L_date"], c="darkred", ls="-.", lw=0.9,
                   label=f"retires {r['L_date'].strftime('%Y-%m')}")
    ax.set_title(f"A{i}", fontsize=9)
    ax.set_ylim(0, 250)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Q2  Rolling 365-day mean permeability (threshold: y=37)", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(ROOT / "Q2输出/figures/fig_q2_rolling_avg.png", dpi=110)
plt.close(fig)
print("Saved fig_q2_rolling_avg.png")

# ---------- Fig 3: 寿命条形图 ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
life_plot = life.sort_values("L_year")
# 替换 inf 为 10+
cap = 10.5
life_plot["L_plot"] = life_plot["L_year"].replace(np.inf, cap)
colors = ["#d9534f" if y < 3 else "#f0ad4e" if y < 7 else "#5cb85c" for y in life_plot["L_plot"]]
bars = ax.barh([f"A{i}" for i in life_plot["i"]],
               life_plot["L_plot"], color=colors, edgecolor="k")
ax.axvline(1, c="k", ls="--", lw=0.5)
ax.axvline(5, c="k", ls="--", lw=0.5)
ax.set_xlabel("Predicted remaining life (years from 2026-01-20)")
ax.set_title("Q2  10-filter life prediction under fixed maintenance rule")
for bar, L, d in zip(bars, life_plot["L_year"], life_plot["L_date"]):
    txt = f">{cap:.0f}y" if np.isinf(L) else f"{L:.2f}y  ({d.strftime('%Y-%m')})"
    ax.text(min(L, cap), bar.get_y()+bar.get_height()/2, "  "+txt,
            va="center", fontsize=9)
ax.set_xlim(0, cap+2)
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(ROOT / "Q2输出/figures/fig_q2_life_bar.png", dpi=130)
plt.close(fig)
print("Saved fig_q2_life_bar.png")
