"""
Q2.5 寿命判定
   L_i = min { d : 滚动365日均值 ȳ_i(d) < 37 }
   另外检查 θ=4.33 判据：该时刻模拟再做一次大维护能否把 ȳ 拉回 ≥37+θ
输出:
   tables/life_prediction.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录(2026_math_modeling_competition)
fut = pd.read_csv(ROOT / "Q2输出/data/future_trajectories.csv", parse_dates=["d"])

# 历史观测 + 未来外推 合并，以便滚动 365 日窗口在过渡期也能计算
hist = pd.read_csv(ROOT / "Q1输出/data/daily_with_vars.csv", parse_dates=["d"])
hist_slim = hist[["i","d","y"]].copy().rename(columns={"y":"y_obs"})
merged = hist_slim.merge(fut[["i","d","y_sim"]], on=["i","d"], how="outer")
# 用观测优先，未观测用模拟
merged["y_mix"] = merged["y_obs"].fillna(merged["y_sim"])
merged = merged.sort_values(["i","d"]).reset_index(drop=True)

# 滚动 365 天均值（每台独立）
merged = merged.sort_values(["i", "d"]).reset_index(drop=True)
merged["rolling_avg_365"] = (
    merged.groupby("i")["y_mix"]
    .transform(lambda s: s.rolling(window=365, min_periods=180).mean())
)
merged.to_csv(ROOT / "Q2输出/data/full_with_rolling.csv", index=False)

# 观测期末的滚动均值（2026-01-19 之前）
cutoff = pd.Timestamp("2026-01-19")
start_future = pd.Timestamp("2026-01-20")

# 寿命：未来段首次 rolling_avg < 37 的日期
rows = []
theta = 4.33
eta_l = 2.25
has_l_bump = 14.68   # from Q1 coeffs — first-time large maintenance adds ~14.68
# Approx improvement for another large maintenance: Δ^l ≈ η_l (short-term pulse)
# If first maintenance hasn't happened yet, additional jump is from has_l; else just η_l
# We'll use η_l + 7 days worth of -ρ_l = -0.21 (approx) = +2.04 net over 7 days
# This is simplified; just use η_l as the representative improvement.

for i in sorted(merged["i"].unique()):
    sub = merged[merged["i"] == i].copy()
    # 只在未来段中找寿命
    fut_sub = sub[sub["d"] >= start_future].reset_index(drop=True)
    below = fut_sub[fut_sub["rolling_avg_365"] < 37]
    if len(below) == 0:
        L_days = np.inf
        L_date = pd.NaT
        L_year = np.inf
        rolling_at_L = fut_sub["rolling_avg_365"].iloc[-1]
    else:
        L_date = below["d"].iloc[0]
        L_days = (L_date - start_future).days
        L_year = L_days / 365.25
        rolling_at_L = below["rolling_avg_365"].iloc[0]

    # 该时刻"再做一次大维护"的预测改善量（用 η_l 近似）
    delta_l_at_L = eta_l  # conservative lower bound — only the H_l7 pulse

    # 起始（2026-01-20）时的滚动均值
    rolling_start = fut_sub["rolling_avg_365"].iloc[0]

    rows.append(dict(
        i=i,
        L_days=L_days,
        L_year=L_year,
        L_date=L_date,
        rolling_avg_at_start=round(rolling_start, 2) if pd.notna(rolling_start) else None,
        rolling_avg_at_L=round(rolling_at_L, 2) if pd.notna(rolling_at_L) else None,
        delta_l_predicted=delta_l_at_L,
        theta=theta,
        retirement_triggered=(L_days != np.inf) and (delta_l_at_L < theta),
    ))

life = pd.DataFrame(rows)
print("10 台寿命预测结果：")
print(life.to_string(index=False))

life.to_csv(ROOT / "Q2输出/tables/life_prediction.csv", index=False)
print(f"\nSaved: Q2输出/tables/life_prediction.csv")
