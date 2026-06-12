# 基于域适应的跨被试脑电情绪识别研究 — 完整实验资料

## 目录结构

```
aresult/
├── report.md                          # 完整实验报告
├── README.md                          # 本文件
├── figures/                           # 实验图表（7张）
│   ├── group_a_subject_dependent.png  # 图1：A组被试内识别准确率
│   ├── comparison_bar.png             # 图2：归一化与域适应对比
│   ├── lambda_sensitivity.png         # 图3：λ敏感性曲线
│   ├── ablation_bar.png               # 图4：消融分析
│   ├── heatmap_per_subject.png         # 图5：各被试准确率热力图
│   ├── confusion_matrices.png          # 图6：混淆矩阵
│   └── tsne_visualization.png         # 图7：t-SNE特征可视化
├── results_v1/                        # v1实验结果JSON（报告所用数据）
│   ├── group_A_subject_dependent_results.json
│   ├── baseline_mixed_lambda1.0_results.json
│   ├── baseline_per_subject_lambda1.0_results.json
│   ├── coral/mmd/dann_*_results.json  (共25个)
│   └── summary_all.csv
└── code/                              # 可复现的实验代码
    ├── README.md                      # 代码说明（环境、依赖、运行命令）
    ├── requirements.txt               # Python依赖
    ├── run_group_a.py                 # A组：被试内识别
    ├── run_loso.py                    # B/C/D/F组：LOSO统一训练
    ├── evaluate_loso.py               # 结果汇总脚本
    ├── visualize_loso.py              # 基础可视化（柱状图、热力图、λ曲线）
    ├── visualize_advanced.py          # 高级可视化（混淆矩阵、t-SNE）
    └── src/                           # 模型与数据源码
        ├── data_loader.py             # 数据加载与预处理
        ├── models/baseline.py         # MLP基线模型
        └── domain_adaptation/
            ├── coral_model.py          # CORAL模型
            ├── coral_loss.py           # CORAL损失
            ├── mmd_model.py            # MMD模型
            ├── mmd_loss.py             # MMD损失
            └── dann_model.py           # DANN模型（含梯度反转层）
```

## 快速开始

```bash
# 1. 安装依赖
cd code/ && pip install -r requirements.txt

# 2. 放置SEED数据集到 code/dataset/dataset/EEG/

# 3. 运行实验（详见 code/README.md）
python run_group_a.py                                         # A组
python run_loso.py --method baseline --norm mixed             # B组
python run_loso.py --method mmd --norm per_subject --lambda_da 10.0  # D组最优

# 4. 生成图表
python visualize_loso.py --fig all
python visualize_advanced.py
```

## 核心结果

| 方法 | 归一化 | λ | 准确率 | 标准差 |
|------|--------|---|--------|--------|
| 被试内 MLP (A组) | per-subject | — | 66.93% | ±14.31% |
| MLP baseline (B组) | mixed | — | 55.20% | ±14.33% |
| MLP baseline (B组) | per-subject | — | 70.73% | ±8.86% |
| CORAL (D组) | per-subject | 5.0 | 74.51% | ±6.63% |
| **MMD (D组)** | **per-subject** | **10.0** | **76.79%** | **±5.85%** |
| DANN (D组) | per-subject | 5.0 | 76.17% | ±6.57% |

