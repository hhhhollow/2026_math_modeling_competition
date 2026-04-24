"""
Q3.4  可视化
  fig1: EAC 热力图 (T_M × T_L) × 10 台
  fig2: 当前 vs 最优 EAC 条形对比
  fig3: 最优维护周期 (T_M*, T_L*) 汇总
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")

grid = pd.read_csv(ROOT / "Q3输出/tables/eac_grid_all.csv")
cmp = pd.read_csv(ROOT / "Q3输出/tables/comparison_fair.csv")
opt = pd.read_csv(ROOT / "Q3输出/tables/optimal_rule_per_filter.csv")

T_M_vals = sorted(grid["T_M"].unique())
T_L_raw = sorted(grid["T_L"].unique())
# T_L: 99999 表示 inf
T_L_labels = [("inf" if v == 99999 else str(v)) for v in T_L_raw]

# ======= FIG 1: 10 台 EAC 热力图 =======
fig, axes = plt.subplots(5, 2, figsize=(13, 18))
for ax, i in zip(axes.ravel(), range(1, 11)):
    sub = grid[grid["i"] == i].copy()
    mat = sub.pivot_table(index="T_M", columns="T_L", values="EAC").reindex(
        index=T_M_vals, columns=T_L_raw)
    im = ax.imshow(mat.values, aspect="auto", origin="lower",
                   cmap="RdYlGn_r", vmin=20, vmax=320)
    ax.set_xticks(range(len(T_L_raw))); ax.set_xticklabels(T_L_labels, fontsize=7)
    ax.set_yticks(range(len(T_M_vals))); ax.set_yticklabels(T_M_vals, fontsize=7)
    # 标出最优点
    best = sub.loc[sub["EAC"].idxmin()]
    xi = T_L_raw.index(best["T_L"])
    yi = T_M_vals.index(best["T_M"])
    ax.plot(xi, yi, marker="*", ms=18, c="blue", mec="white", mew=1.5)
    ax.set_xlabel("T_L (d)", fontsize=8)
    ax.set_ylabel("T_M (d)", fontsize=8)
    ax.set_title(f"A{i}   EAC* = {best['EAC']:.1f}  at (T_M={int(best['T_M'])}, T_L={T_L_labels[xi]})",
                 fontsize=9)
    # 在每格写数值
    for r, T_M in enumerate(T_M_vals):
        for c, T_L in enumerate(T_L_raw):
            v = mat.iloc[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 150 else "black")
fig.suptitle("Q3  EAC heatmap (unit: 10k CNY/yr)   * = optimal",
             fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(ROOT / "Q3输出/figures/fig_q3_eac_heatmap.png", dpi=110)
plt.close(fig)
print("Saved fig_q3_eac_heatmap.png")

# ======= FIG 2: 当前 vs 最优 EAC 条形对比 =======
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cmp))
w = 0.35
ax.bar(x - w/2, cmp["EAC_cur"], w, label="Current rule (Q2, 20y horizon)",
       color="#7f7f7f", edgecolor="k")
ax.bar(x + w/2, cmp["EAC_opt"], w, label="Optimal rule (Q3)",
       color="#2ca02c", edgecolor="k")
for xi, (cur, opt_v, pct) in enumerate(zip(cmp["EAC_cur"], cmp["EAC_opt"], cmp["save_pct"])):
    color = "darkgreen" if pct > 0 else "darkred"
    ax.text(xi, max(cur, opt_v) + 8, f"{pct:+.1f}%",
            ha="center", fontsize=8, color=color, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"A{i}" for i in cmp["i"]])
ax.set_ylabel("EAC (10k CNY/yr)")
ax.set_title("Q3  Current vs Optimal maintenance rule — annual equivalent cost")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")
# 底部加 A4/A6 警示
ax.text(0.02, -0.15, "⚠ A4 / A6 have positive historical β — their 'no-retirement' is extrapolation artifact; "
                    "EAC values are lower bounds under the 20-year horizon.",
        transform=ax.transAxes, fontsize=8, color="darkred", style="italic")
fig.tight_layout()
fig.savefig(ROOT / "Q3输出/figures/fig_q3_eac_comparison.png", dpi=130,
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_q3_eac_comparison.png")

# ======= FIG 3: 最优 (T_M*, T_L*) 规划图 =======
fig, ax = plt.subplots(figsize=(11, 5))
# 堆积条形：T_M 和 T_L 两根
i_order = opt["i"].values
width = 0.4
x = np.arange(len(opt))
# T_M bar
bars_m = ax.bar(x - width/2, opt["T_M"], width, label="T_M* (medium cycle)",
                color="#1f77b4", edgecolor="k")
# T_L bar (cap inf to 800 for plotting)
T_L_plot = [800 if (isinstance(s, float) and np.isinf(s)) or str(s)=="inf" else float(s)
            for s in opt["T_L_label"]]
tl_is_inf = [(isinstance(s, float) and np.isinf(s)) or str(s)=="inf"
             for s in opt["T_L_label"]]
bars_l = ax.bar(x + width/2, T_L_plot, width, label="T_L* (large cycle)",
                color=["#d62728" if infl else "#ff7f0e" for infl in tl_is_inf],
                edgecolor="k")
for xi, (tm, tl_v, infl) in enumerate(zip(opt["T_M"], T_L_plot, tl_is_inf)):
    ax.text(xi - width/2, tm + 15, f"{int(tm)}", ha="center", fontsize=9)
    label = "∞ (no large)" if infl else f"{int(tl_v)}"
    ax.text(xi + width/2, tl_v + 15, label,
            ha="center", fontsize=8, color="darkred" if infl else "black")
ax.set_xticks(x); ax.set_xticklabels([f"A{i}" for i in i_order])
ax.set_ylabel("days")
ax.set_title("Q3  Optimal maintenance periods   (T_L='inf' plotted at 800 for display)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 900)
fig.tight_layout()
fig.savefig(ROOT / "Q3输出/figures/fig_q3_optimal_periods.png", dpi=130)
plt.close(fig)
print("Saved fig_q3_optimal_periods.png")
