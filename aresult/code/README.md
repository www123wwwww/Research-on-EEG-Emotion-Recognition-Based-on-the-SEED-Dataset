# 基于域适应的跨被试脑电情绪识别研究

## 运行环境

- Python 3.10+
- CUDA（可选，GPU加速训练）
- 依赖库见 `code/requirements.txt`

## 安装依赖

```bash
cd code/
pip install -r requirements.txt
```

## 数据准备

本实验使用SEED数据集（http://bcmi.sjtu.edu.cn/~seed/）。请将数据集解压后放置于以下目录结构：

```
code/
├── dataset/
│   └── dataset/
│       └── EEG/
│           ├── 1_1.npz
│           ├── 1_2.npz
│           ├── 1_3.npz
│           ├── 2_1.npz
│           └── ...    # 共 36 个 .npz 文件（12被试 × 3 session）
├── src/
│   ├── data_loader.py
│   ├── models/
│   │   └── baseline.py
│   └── domain_adaptation/
│       ├── coral_model.py
│       ├── coral_loss.py
│       ├── mmd_model.py
│       ├── mmd_loss.py
│       └── dann_model.py
├── run_group_a.py
├── run_loso.py
├── evaluate_loso.py
├── visualize_loso.py
├── visualize_advanced.py
└── requirements.txt
```

每个 `.npz` 文件包含：
- `train_data`: pickle序列化的字典，键为5个频段名（delta/theta/alpha/beta/gamma），值为 (N, 62) 数组
- `test_data`: 同上
- `train_label`: (N,) int64，0=消极/1=中性/2=积极
- `test_label`: 同上

### 数据下载

SEED数据集需从原作者网站申请下载。下载后运行以下命令验证：

```bash
cd code/
python -c "from src.data_loader import print_dataset_summary; print_dataset_summary()"
```

## 实验超参数（统一）

| 参数 | 值 |
|------|-----|
| 最大训练轮数 | 20 |
| 学习率 | 5e-4 |
| 批大小 | 128 |
| 权重衰减 | 1e-4 |
| 随机种子 | 42 |
| 优化器 | Adam |
| 学习率调度 | CosineAnnealingLR |
| 早停 | 无 |
| 验证集划分 | 时序划分（前80%训练/后20%验证，不打乱顺序） |
| 模型 | 3层MLP (310→256→128→3)，BatchNorm+Dropout(0.3) |

## 运行命令

### A组：被试内识别基线

```bash
python run_group_a.py
```

结果保存至 `results/group_A_subject_dependent_results.json`。

### B组：LOSO基线

```bash
python run_loso.py --method baseline --norm mixed          # 混合归一化基线
python run_loso.py --method baseline --norm per_subject    # 被试内归一化基线
```

### C组：混合归一化+域适应（λ=1.0）

```bash
python run_loso.py --method baseline --norm mixed
python run_loso.py --method coral --norm mixed --lambda_da 1.0
python run_loso.py --method mmd --norm mixed --lambda_da 1.0
python run_loso.py --method dann --norm mixed --lambda_da 1.0
```

### D组：被试内归一化+域适应（最优λ）

```bash
python run_loso.py --method coral --norm per_subject --lambda_da 5.0
python run_loso.py --method mmd --norm per_subject --lambda_da 10.0
python run_loso.py --method dann --norm per_subject --lambda_da 5.0
```

### F组：λ敏感性分析

```bash
# CORAL
for lam in 0.1 0.5 1.0 2.0 5.0 10.0 20.0; do
    python run_loso.py --method coral --norm per_subject --lambda_da $lam
done

# MMD
for lam in 0.1 0.5 1.0 2.0 5.0 10.0 20.0; do
    python run_loso.py --method mmd --norm per_subject --lambda_da $lam
done

# DANN
for lam in 0.1 0.5 1.0 2.0 5.0 10.0 20.0; do
    python run_loso.py --method dann --norm per_subject --lambda_da $lam
done
```

### 结果汇总与可视化

```bash
python evaluate_loso.py          # 生成 summary_all.csv
python visualize_loso.py --fig all    # 生成所有图表
python visualize_advanced.py    # 生成混淆矩阵和t-SNE可视化
```

图表保存至 `../aresult/figures/`。

## 结果文件说明

所有实验结果（JSON格式）保存在 `../aresult/results_v1/` 目录：

| 文件 | 说明 |
|------|------|
| `group_A_subject_dependent_results.json` | A组：被试内识别基线 |
| `baseline_mixed_lambda1.0_results.json` | B组：混合归一化基线 |
| `baseline_per_subject_lambda1.0_results.json` | B组：被试内归一化基线 |
| `coral_mixed_lambda1.0_results.json` | C组：混合归一化+CORAL |
| `mmd_mixed_lambda1.0_results.json` | C组：混合归一化+MMD |
| `dann_mixed_lambda1.0_results.json` | C组：混合归一化+DANN |
| `coral_per_subject_lambda*.0_results.json` | F组：CORAL λ扫描 |
| `mmd_per_subject_lambda*.0_results.json` | F组：MMD λ扫描 |
| `dann_per_subject_lambda*.0_results.json` | F组：DANN λ扫描 |
| `summary_all.csv` | 汇总表格 |

## 开源代码来源与修改说明

本项目基于开源代码进行开发，原始代码来源于：

**原始仓库**：`Research-on-EEG-Emotion-Recognition-Based-on-the-SEED-Dataset`
- 原始实现包含：SVM基线、MLP基线、CORAL和MMD域适应方法（混合归一化，λ=1.0）
- 原始LOSO结果：CORAL 66.06%，MMD 67.45%（混合归一化，λ=1.0）

**本组的修改和新增内容**：

1. **新增DANN方法**：实现了域对抗神经网络（`dann_model.py`），包含梯度反转层
2. **新增被试内归一化**：在 `data_loader.py` 中新增 `get_loso_splits_subject_norm()` 函数，实现per-subject z-score归一化
3. **统一训练脚本**：重写 `run_loso.py`，统一所有方法的训练流程和超参数
4. **新增A组实验**：`run_group_a.py` 实现被试内subject-dependent基线
5. **扩展λ扫描**：从原始λ=1.0扩展到λ∈{0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0}
6. **统一超参数**：所有实验使用相同的epoch=20, lr=5e-4, batch_size=128，去除早期停止
7. **可视化脚本**：新增 `visualize_loso.py` 和 `visualize_advanced.py`，生成论文所有图表
8. **评估脚本**：新增 `evaluate_loso.py`，自动汇总所有实验结果

## 实验报告

完整实验报告见 `../aresult/report.md`，包含：

- A组：被试内识别基线（66.93% ± 14.31%）
- B组：跨被试LOSO基线（混合55.20%，被试内70.73%）
- C组：混合归一化+域适应对比
- D组：被试内归一化+最优λ对比（MMD λ=10达76.79%）
- F组：λ敏感性分析（7个λ值×3种方法）
- 消融分析、混淆矩阵、t-SNE可视化

## 核心结果

| 方法 | 归一化 | λ | 准确率 | 标准差 |
|------|--------|---|--------|--------|
| 被试内 MLP (A组) | per-subject | — | 66.93% | ±14.31% |
| MLP baseline (B组) | mixed | — | 55.20% | ±14.33% |
| MLP baseline (B组) | per-subject | — | 70.73% | ±8.86% |
| CORAL (D组) | per-subject | 5.0 | 74.51% | ±6.63% |
| MMD (D组) | per-subject | 10.0 | **76.79%** | ±5.85% |
| DANN (D组) | per-subject | 5.0 | 76.17% | ±6.57% |