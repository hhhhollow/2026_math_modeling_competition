# 2026 Mathematical Modeling Competition

Solution for the 2026 Beijing Collegiate Mathematical Modeling Contest problem on **filter equipment monitoring, maintenance modeling, and lifecycle optimization**. The project covers data preprocessing, regression analysis, life prediction, cost optimization, and sensitivity analysis across four sub-problems (Q1–Q4), together with the LaTeX paper.

## Project Structure

```
2026_math_modeling_competition/
├── 题目/                  # Problem statement and raw data
│   ├── 设备检测.docx          # Problem description
│   ├── 附件1.xlsx             # Raw permeability measurements
│   └── 附件2.xlsx             # Maintenance records
├── Q1输出/                # Q1: data processing & pattern analysis
│   ├── code/                  # step1–step8 Python scripts
│   ├── data/                  # intermediate / cleaned data
│   ├── figures/               # plots
│   ├── tables/                # result tables (CSV)
│   └── Q1_summary.md          # results summary
├── Q2输出/                # Q2: maintenance rule + life prediction
├── Q3输出/                # Q3: EAC-based optimal maintenance plan
├── Q4输出/                # Q4: sensitivity & elasticity analysis
├── lym/                   # Jupyter notebooks (author: lym)
├── lyx/Jupyter复现/       # Reproducible notebooks (Q1–Q4)
├── 论文/                  # LaTeX paper
│   ├── main.tex               # main document (xelatex)
│   ├── main.pdf               # compiled output
│   ├── figures/               # figures used in the paper
│   └── README.md              # build instructions
└── .venv/                 # Python virtual environment
```

## Sub-problems Overview

| Q | Topic | Key result |
|---|-------|-----------|
| Q1 | Daily aggregation, missing/outlier handling, fixed-effects OLS/WLS regression with maintenance windows and seasonality | OLS R² = 0.8385 (n = 5046, k = 28); A4/A6 anomalies isolated |
| Q2 | Fixed maintenance rule extraction, CV-tuned long-term damage γ, linear vs. exponential degradation, forward simulation & life prediction | Linear model preferred; per-filter remaining-life table |
| Q3 | Equivalent Annual Cost (EAC) grid search; optimal vs. current maintenance policy over 20-year horizon | ~15.0% cost saving over the current rule |
| Q4 | Single-parameter sweep, switching point & elasticity, joint sensitivity, extreme-scenario stress test | Tornado chart of cost drivers; robust optimum |

## Environment

- Python 3 (`.venv/` is the local virtual environment)
- Typical dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`, `jupyter`
- LaTeX: TeX Live or MacTeX, compiled with **xelatex** (Chinese support required)

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running the Pipelines

Each `Qx输出/code/` folder contains numbered scripts that should be run in order. Example for Q1:

```bash
cd Q1输出/code
python step1_load_merge.py
python step2_daily_agg.py
python step3_missing_outlier.py
python step4_maintenance_vars.py
python step5_stl.py
python step6_ols_wls.py
python step7_R_indicator.py
python step8_figures.py
```

Q2, Q3, Q4 follow the same `stepN_*.py` convention. Reproducible notebook versions are available in [lyx/Jupyter复现/](lyx/Jupyter复现/).

## Building the Paper

```bash
cd 论文
xelatex main.tex
xelatex main.tex   # second pass for TOC and cross-references
```

Or with `latexmk`:

```bash
latexmk -xelatex main.tex
```

See [论文/README.md](论文/README.md) for paper-specific notes.

## Notes

- Permeability values are stored as **index values** (the `%` sign is dropped; the maximum in 附件1 reaches 180).
- 附件2 contains only `{m, l}` maintenance types (110 medium + 17 large = 127 records).
- A 82–83-day continuous gap (2026-01-20 to 2026-04-11) reflects a sampling-frequency drop at the plant, not equipment failure; the main analysis uses the dense 2024-04 to 2026-01 window.
