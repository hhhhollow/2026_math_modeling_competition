# 2026 数学建模竞赛

本仓库为 2026 北京高校数学建模校际联赛 A 题“过滤设备监测”解题工程，主题为**过滤设备透水率监测、维护建模与生命周期优化**。项目围绕 Q1--Q4 四个子问题展开，包含数据预处理、固定效应回归、寿命预测、维护成本优化、敏感性分析，以及最终 LaTeX 论文。

## 项目结构

```
2026_math_modeling_competition/
├── 题目/                  # 题目文件与原始数据
│   ├── 设备检测.docx          # 题目说明
│   ├── 附件1.xlsx             # 原始小时级透水率数据
│   └── 附件2.xlsx             # 维护记录
├── Q1输出/                # Q1：数据处理与规律识别
│   ├── code/                  # step1--step8 Python 脚本
│   ├── data/                  # 中间数据与清洗后数据
│   ├── figures/               # 可视化图
│   ├── tables/                # 结果表格 CSV
│   └── Q1_summary.md          # Q1 结果摘要
├── Q2输出/                # Q2：当前维护规律提取与寿命预测
├── Q3输出/                # Q3：基于 EAC 的最优维护方案
├── Q4输出/                # Q4：成本敏感性与弹性分析
├── lym/                   # Jupyter 笔记本
├── lyx/Jupyter复现/       # Q1--Q4 可复现笔记本
├── 论文/                  # LaTeX 论文
│   ├── main.tex               # 主文档，使用 xelatex 编译
│   ├── main.pdf               # 编译后的论文
│   ├── figures/               # 论文插图
│   └── README.md              # 论文编译说明
└── .venv/                 # Python 本地虚拟环境
```

## 子问题概览

| 子问题 | 主要内容 | 关键结果 |
|---|---|---|
| Q1 | 日中位聚合、缺失与异常处理、维护窗口构造、固定效应 OLS/WLS 回归、季节性与维护效应分析 | OLS 全样本 $R^2=0.8385$，有效样本 5046 行，模型 28 个参数；识别 A4/A6 启动期异常 |
| Q2 | 当前固定维护规律提取、累积维护项交叉验证、线性/指数衰减比较、向前仿真与寿命预测 | 线性模型优于指数模型；输出 10 台过滤器寿命预测表 |
| Q3 | 等值年化成本（EAC）建模，在 $(T_M,T_L)$ 网格上搜索最优维护周期 | 12 年地平线下，最优方案较当前规律合计节省约 213.7 万元/年，节省率约 14.7% |
| Q4 | 成本参数 $\pm 30\%$ 扫描、弹性分析、联合扰动与极端场景测试 | 购置成本 $c_{\text{buy}}$ 为主导驱动因素，维护方案整体较稳健 |

## 运行环境

- Python 3（仓库内 `.venv/` 为本地虚拟环境）
- 常用依赖：`numpy`、`pandas`、`scipy`、`statsmodels`、`matplotlib`、`openpyxl`、`jupyter`
- LaTeX：TeX Live 或 MacTeX，使用 **xelatex** 编译，需要中文支持

激活虚拟环境：

```bash
source .venv/bin/activate
```

## 运行数据处理流程

各 `Qx输出/code/` 目录下均包含按步骤编号的脚本，建议按顺序运行。以 Q1 为例：

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

Q2、Q3、Q4 同样采用 `stepN_*.py` 的脚本命名方式。可复现的 Jupyter 版本见 [lyx/Jupyter复现/](lyx/Jupyter复现/)。

## 编译论文

```bash
cd 论文
xelatex main.tex
xelatex main.tex   # 第二遍用于目录和交叉引用
```

也可以使用 `latexmk`：

```bash
latexmk -xelatex main.tex
```

论文相关说明见 [论文/README.md](论文/README.md)。

## 说明

- 透水率以**指数值**形式保存，数据中未保留 `%` 符号；附件 1 中最大值可达 180。
- 附件 2 仅包含 `{m, l}` 两类维护记录，共 127 条，其中中维护 110 条、大维护 17 条。
- 2026-01-20 至 2026-04-11 附近存在约 82--83 天连续稀疏区间，主要反映厂方采样频率下降，并非设备故障；主分析重点使用 2024-04 至 2026-01 的密集观测区间。
