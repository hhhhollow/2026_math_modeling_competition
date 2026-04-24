"""
Q1.8 整合图表（EN titles to avoid CJK font issues）
  fig3  R boxplot by type×season
  fig4  trend β_i per filter
  fig5  fitted vs observed
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Q1输出"
R = pd.read_csv(ROOT / "tables/R_events.csv")
trend = pd.read_csv(ROOT / "tables/trend_per_filter.csv")
daily = pd.read_csv(ROOT / "data/daily_with_vars.csv", parse_dates=["d"])
resid = pd.read_csv(ROOT / "data/regression_residuals.csv", parse_dates=["d"])

# map Chinese season to English
SMAP = {"春": "Spring", "夏": "Summer", "秋": "Autumn", "冬": "Winter"}
R["season_en"] = R["season"].map(SMAP)

# --- Fig 3: R boxplot by type×season ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
seasons = ["Spring", "Summer", "Autumn", "Winter"]
for ax, q, lab in zip(axes, ["m", "l"], ["Medium maintenance", "Large maintenance"]):
    dat = [R[(R["q"] == q) & (R["season_en"] == s)]["R"].values for s in seasons]
    ax.boxplot(dat, labels=seasons, showmeans=True, meanline=True)
    ax.axhline(0, c="grey", lw=0.7, ls="--")
    ax.set_title(f"{lab}  (n={(R['q']==q).sum()})")
    ax.set_ylabel("R = mean(y_post7) - mean(y_pre7)")
    ax.grid(alpha=0.3)
fig.suptitle("Core indicator R^{q,7}: post - pre 7-day permeability delta by type x season",
             fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "figures/fig3_R_boxplot.png", dpi=130)
plt.close(fig)

# --- Fig 4: trend per filter ---
fig, ax = plt.subplots(figsize=(9, 4.5))
trend_sorted = trend.sort_values("beta_per_year")
colors = ["#d9534f" if v < 0 else "#5cb85c" for v in trend_sorted["beta_per_year"]]
bars = ax.barh([f"A{i}" for i in trend_sorted["i"]],
               trend_sorted["beta_per_year"], color=colors, edgecolor="k")
ax.axvline(0, c="k", lw=0.7)
ax.set_xlabel("Annualized trend slope  (permeability index / year)")
ax.set_title("Filter-specific natural aging slope (OLS estimate)")
for bar, v in zip(bars, trend_sorted["beta_per_year"]):
    ax.text(v, bar.get_y()+bar.get_height()/2,
            f" {v:+.1f}", va="center", fontsize=9,
            ha="left" if v >= 0 else "right")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(ROOT / "figures/fig4_trend_per_filter.png", dpi=130)
plt.close(fig)

# --- Fig 5: fitted vs observed ---
daily2 = daily.dropna(subset=["y"]).copy()
daily2 = daily2.merge(resid[["i","d","resid_ols"]], on=["i","d"])
daily2["y_hat"] = daily2["y"] - daily2["resid_ols"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(daily2["y"], daily2["y_hat"], s=2, alpha=0.3, c="steelblue")
lim = [daily2["y"].min()-5, daily2["y"].max()+5]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed y")
ax.set_ylabel("Fitted y_hat")
r2 = 1 - (daily2["resid_ols"]**2).sum() / ((daily2["y"]-daily2["y"].mean())**2).sum()
ax.set_title(f"OLS fitted vs observed    R^2 = {r2:.3f},  n = {len(daily2)}")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "figures/fig5_fit_vs_observed.png", dpi=130)
plt.close(fig)

print("Figures saved:")
for f in ["fig3_R_boxplot", "fig4_trend_per_filter", "fig5_fit_vs_observed"]:
    print("  ", ROOT / f"figures/{f}.png")

