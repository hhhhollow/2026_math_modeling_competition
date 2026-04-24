# 2026 数学建模比赛项目

本文件夹用于整理数学建模比赛的数据、代码、论文材料和 Jupyter Notebook。

## 目录

- `A题/`: 题目附件与原始数据
- `Q1输出/`: 第一问代码、表格、图片和结果摘要
- `notebooks/`: Jupyter Notebook 工作区
- `scripts/`: 可复用脚本
- `docs/`: 论文、说明文档和过程记录

## 启动 Jupyter

```bash
cd "/Users/hhhhollow/Desktop/2026_math_modeling_competition"
./run_jupyter.sh
```

Jupyter Lab 启动后，Notebook 内核选择 `Python (2026 Math Modeling)`。

## 依赖

项目依赖写在 `requirements.txt` 中。当前 Q1 代码主要需要：

- `pandas`
- `numpy`
- `matplotlib`
- `openpyxl`
- `jupyterlab`
- `notebook`
- `ipykernel`
- `pyarrow`
