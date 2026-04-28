"""
Q1.1 读取附件1并合并为长表
输出: data/hourly_long.parquet  (columns: i, h, p)
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "题目" / "附件1.xlsx"
OUT = PROJECT_ROOT / "Q1输出" / "data"
OUT.mkdir(parents=True, exist_ok=True)

xl = pd.ExcelFile(SRC, engine="openpyxl")
print("Sheets:", xl.sheet_names)

frames = []
for sn in xl.sheet_names:
    df = xl.parse(sn)
    # 规范列名
    df.columns = [c.strip().lower() for c in df.columns]
    assert set(df.columns) >= {"time", "per"}, f"{sn} columns: {df.columns}"
    # 提取编号 i
    i = int(sn.split("_")[1])
    df = df.rename(columns={"time": "h", "per": "p"})[["h", "p"]]
    df["i"] = i
    df["h"] = pd.to_datetime(df["h"])
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    frames.append(df)
    print(f"  A_{i}: rows={len(df)}, p>100 count={(df['p']>100).sum()}, "
          f"p<0 count={(df['p']<0).sum()}, NaN={df['p'].isna().sum()}, "
          f"range=[{df['p'].min():.2f}, {df['p'].max():.2f}]")

long = pd.concat(frames, ignore_index=True).sort_values(["i", "h"]).reset_index(drop=True)
print(f"\nTotal rows: {len(long)}, filters: {long['i'].nunique()}")
print(f"Time span: {long['h'].min()} .. {long['h'].max()}")

# 保存
long.to_parquet(OUT / "hourly_long.parquet", index=False) if False else None
long.to_csv(OUT / "hourly_long.csv", index=False)
print(f"Saved: {OUT/'hourly_long.csv'}")

# 简要统计
stats = long.groupby("i")["p"].agg(["count", "min", "max", "mean", "median"]).round(2)
stats["nan_count"] = long.groupby("i")["p"].apply(lambda s: s.isna().sum())
print("\nPer-filter summary:")
print(stats)
stats.to_csv(OUT / "per_filter_hourly_stats.csv")
