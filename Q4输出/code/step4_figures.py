"""
Q4.4 图表生成
  fig1: tornado chart 全队 EAC 对三个成本参数的敏感度
  fig2: per-filter EAC 弹性条形图
  fig3: c_buy × c_l 联合热力图
  fig4: EAC vs c_buy 曲线（10 台叠加）
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/sessions/serene-cool-hawking/mnt/2026_math_modeling_competition")

fleet = pd.read_csv(ROOT / "Q4输出/tables/sensitivity_total_fleet.csv")
sweep = pd.read_csv(ROOT / "Q4输出/tables/sensitivity_single_param.csv")
elast = pd.read_csv(ROOT / "Q4输出/tables/per_filter_elasticity.csv")
joint = pd.read_csv(ROOT / "Q4输出/tables/joint_buy_l.csv")

base_eac = fleet[(fleet.param == "c_buy") & (fleet.level == 0)]["EAC_total"].iloc[0]

# ===== Fig 1: Tornado chart =====
fig, ax = plt.subplots(figsize=(9, 4))
params = ["c_buy", "c_m", "c_l"]
labels = ["$c_{buy}$ (300 wY)", "$c_m$ (3 wY)", "$c_l$ (12 wY)"]
y_pos = np.arange(len(params))
low_vals, high_vals = [], []
for p in params:
    sub = fleet[fleet.param == p].sort_values("level")
    low_vals.append(sub["EAC_total"].iloc[0] - base_eac)   # -30%
    high_vals.append(sub["EAC_total"].iloc[-1] - base_eac) # +30%

ax.barh(y_pos, low_vals, color="#1f77b4", edgecolor="k", label="-30%")
ax.barh(y_pos, high_vals, color="#d62728", edgecolor="k", label="+30%")
ax.axvline(0, color="k", lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
for i, (lo, hi) in enumerate(zip(low_vals, high_vals)):
    ax.text(lo - 5, i, f"{lo:+.0f}", va="center", ha="right", fontsize=9)
    ax.text(hi + 5, i, f"{hi:+.0f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("Total fleet EAC change (10k CNY/yr)")
ax.set_title(f"Q4  Tornado: fleet EAC sensitivity to ±30% cost shocks\n"
             f"baseline = {base_eac:.1f} 10k CNY/yr")
ax.legend(loc="lower right")
ax.grid(alpha=0.3, axis="x")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(ROOT / "Q4输出/figures/fig_q4_tornado.png", dpi=130)
plt.close(fig)
print("Saved fig_q4_tornado.png")

# ===== Fig 2: per-filter elasticity =====
fig, ax = plt.subplots(figsize=(11, 5))
i_arr = elast["i"].values
x = np.arange(len(i_arr))
w = 0.27
ax.bar(x - w, elast["eps_c_buy"], w, label="ε to $c_{buy}$", color="#1f77b4", edgecolor="k")
ax.bar(x,     elast["eps_c_m"],   w, label="ε to $c_m$",    color="#2ca02c", edgecolor="k")
ax.bar(x + w, elast["eps_c_l"],   w, label="ε to $c_l$",    color="#ff7f0e", edgecolor="k")
ax.set_xticks(x); ax.set_xticklabels([f"A{i}" for i in i_arr])
ax.set_ylabel("Elasticity ε = (dEAC/EAC) / (dc/c)")
ax.set_title("Q4  Per-filter EAC* elasticity to cost parameters")
ax.legend()
ax.grid(alpha=0.3, axis="y")
ax.axhline(1, color="k", ls="--", lw=0.5)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(ROOT / "Q4输出/figures/fig_q4_elasticity.png", dpi=130)
plt.close(fig)
print("Saved fig_q4_elasticity.png")

# ===== Fig 3: joint heatmap c_buy × c_l =====
fig, ax = plt.subplots(figsize=(9, 6.5))
piv = joint.pivot_table(index="level_buy", columns="level_l", values="EAC_total")
levs_buy = sorted(joint["level_buy"].unique())
levs_l = sorted(joint["level_l"].unique())
mat = piv.reindex(index=levs_buy, columns=levs_l).values
im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdYlGn_r")
ax.set_xticks(range(len(levs_l)))
ax.set_xticklabels([f"{int(l*100):+d}%" for l in levs_l])
ax.set_yticks(range(len(levs_buy)))
ax.set_yticklabels([f"{int(l*100):+d}%" for l in levs_buy])
ax.set_xlabel("$c_l$ deviation from baseline (12 10k CNY)")
ax.set_ylabel("$c_{buy}$ deviation from baseline (300 10k CNY)")
ax.set_title("Q4  Total fleet EAC under joint cost shocks   ($c_m$=3 fixed)")
for r in range(len(levs_buy)):
    for c in range(len(levs_l)):
        v = mat[r, c]
        ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                fontsize=8, color="white" if v > 1100 or v < 800 else "black")
plt.colorbar(im, ax=ax, label="EAC total (10k CNY/yr)")
fig.tight_layout()
fig.savefig(ROOT / "Q4输出/figures/fig_q4_joint_heatmap.png", dpi=130)
plt.close(fig)
print("Saved fig_q4_joint_heatmap.png")

# ===== Fig 4: EAC vs c_buy curves per filter =====
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for k, i in enumerate(range(1, 11)):
    sub = sweep[(sweep.i == i) & (sweep.param == "c_buy")].sort_values("level")
    ax.plot(sub["level"]*100, sub["EAC"], marker="o", lw=1.5,
            label=f"A{i}", color=colors[k])
ax.set_xlabel("$c_{buy}$ deviation from baseline (%)")
ax.set_ylabel("Optimal EAC* per filter (10k CNY/yr)")
ax.set_title("Q4  Optimal EAC* vs $c_{buy}$ — by filter")
ax.legend(ncol=2, loc="center left", fontsize=9)
ax.grid(alpha=0.3)
ax.axvline(0, color="k", lw=0.5, ls=":")
fig.tight_layout()
fig.savefig(ROOT / "Q4输出/figures/fig_q4_eac_vs_cbuy.png", dpi=130)
plt.close(fig)
print("Saved fig_q4_eac_vs_cbuy.png")
