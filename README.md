# EEG Emotion Recognition on SEED Dataset

基于 SEED 数据集的脑电情绪识别研究，涵盖**注意力机制、跨 session 迁移学习（含 session 间域适应）与跨被试域适应**四个研究方向。

> 研究思路与每步发现见 [RESEARCH_STORY.md](RESEARCH_STORY.md)

---

## 完整实验结果

### Subject-Dependent（被试依赖）

| 模型 | 准确率 | vs MLP 基线 |
|------|:------:|:-----------:|
| SVM（baseline） | 71.40% ± 16.63% | +3.33% |
| MLP（baseline） | 68.07% ± 18.69% | — |
| MLP + Attention | 65.32% ± 16.70% | −2.75% |
| SVM + Cross-Session | 78.17% ± 14.55% | +10.10% |
| MLP + Cross-Session | 77.63% ± 14.22% | +9.56% |
| MLP + CS + AdaBN（无微调） | 80.28% ± 14.31% | +12.21% |
| MLP + CS + AdaBN + Finetune | 80.46% ± 13.73% | +12.39% |
| **MLP + CS + CORAL** | **82.09% ± 10.78%** | **+14.02%** |

### LOSO（跨被试，Leave-One-Subject-Out）

| 模型 | 准确率 | vs MLP 基线 |
|------|:------:|:-----------:|
| SVM（baseline） | 53.60% ± 12.20% | — |
| MLP（baseline） | 56.90% ± 13.17% | — |
| MLP + Attention | 55.27% ± 14.55% | −1.63% |
| MLP + CORAL | 66.06% ± 12.65% | +9.16% |
| MLP + DANN | 66.79% ± 12.23% | +9.89% |
| **MLP + MMD** | **67.45%** ± 11.83% | **+10.55%** |

---

## 项目结构

```
├── README.md                   # 本文件：项目说明与结果总览
├── RESEARCH_STORY.md           # 研究思路、发现与结论的完整叙述
├── requirements.txt
├── dataset/dataset/
│   ├── EEG/                    # {subject}_{session}.npz，共 36 个文件（12 被试 × 3 session）
│   └── EYE/                    # 眼动特征（本研究未使用）
│
├── src/
│   ├── data_loader.py          # 数据读取 + 两种划分策略（Subject-Dependent / LOSO）
│   ├── train.py                # 基线训练主入口（SVM / MLP / MLP+Attention）
│   ├── evaluate.py             # 基线结果汇总表格
│   ├── visualize.py            # 注意力权重可视化（频段柱状图 + 电极热力图）
│   │
│   ├── models/
│   │   ├── baseline.py         # SVMModel（RBF核）+ MLPModel（310→256→128→3）
│   │   └── attention_mlp.py    # AttentionMLP：SE风格频段注意力 + 通道注意力 + MLP分类头
│   │
│   ├── domain_adaptation/      # 跨被试域适应（LOSO场景）
│   │   ├── coral_loss.py       # CORAL损失函数（对齐协方差矩阵）
│   │   ├── coral_model.py      # CoralMLP：暴露128维中间特征的MLP
│   │   ├── train_coral.py      # LOSO训练：MLP + CORAL loss
│   │   ├── evaluate_coral.py   # 对比表：SVM/MLP/Attention/CORAL
│   │   ├── visualize_coral.py  # CORAL结果柱状图
│   │   │
│   │   ├── mmd_loss.py         # MMD损失函数（多核RBF，5种带宽）
│   │   ├── mmd_model.py        # MmdMLP：结构同CoralMLP，独立文件
│   │   ├── train_mmd.py        # LOSO训练：MLP + MMD loss
│   │   ├── evaluate_mmd.py     # 对比表：5种方法（含CORAL）
│   │   ├── visualize_mmd.py    # 5种方法柱状图 + 逐折delta折线图
│   │   │
│   │   ├── dann_model.py       # DannMLP：梯度反转层（GRL）+ 情绪/域双分类头
│   │   ├── train_dann.py       # LOSO训练：MLP + DANN对抗训练（α调度：0→1）
│   │   ├── evaluate_dann.py    # 对比表：6种方法（含CORAL/MMD/DANN）
│   │   └── visualize_dann.py   # 6种方法柱状图 + 逐折delta折线图（最终综合图）
│   │
│   └── cross_session/          # 跨session迁移学习 + session间域适应（Subject-Dependent场景）
│       ├── train_cross_session.py             # MLP：其他2个session预训练 → 目标session微调
│       ├── train_svm_cross_session.py         # SVM：合并其他2个session数据一次性训练
│       ├── train_cross_session_coral.py       # MLP + 预训练时加CORAL对齐session分布
│       ├── train_cross_session_adabn.py       # MLP + 预训练后用AdaBN替换BN统计量
│       ├── evaluate_cross_session.py          # 对比表：基线3种 + MLP+跨session
│       ├── evaluate_full.py                   # 对比表：全部5种方法（含SVM+跨session）
│       ├── evaluate_cross_session_coral.py    # 对比表：含CORAL方法
│       ├── evaluate_cross_session_adabn.py    # 对比表：含AdaBN变体（无FT vs FT）
│       ├── visualize_cross_session.py         # MLP跨session的柱状图 + 逐折折线图
│       ├── visualize_full.py                  # 全部5种方法：柱状图 + delta图
│       ├── visualize_cross_session_coral.py   # CORAL结果：柱状图 + CORAL贡献delta图
│       └── visualize_cross_session_adabn.py   # AdaBN结果：柱状图 + 零标签免费增益delta图
│
└── results/
    ├── summary.csv                         # 基线6种实验汇总（自动生成）
    ├── *_svm_results.npy                   # SVM各场景结果
    ├── *_mlp_results.npy                   # MLP各场景结果
    ├── *_attn_mlp_results.npy             # MLP+Attention各场景结果
    ├── band_attention_{split}.png          # 5频段注意力权重柱状图
    ├── channel_attention_{split}.png       # 62通道脑电帽注意力热力图
    ├── accuracy_comparison.png             # 基线3模型对比图
    │
    ├── domain_adaptation/                  # 域适应实验结果
    │   ├── loso_mlp_coral_results.npy
    │   ├── loso_mlp_mmd_results.npy
    │   ├── loso_mlp_dann_results.npy
    │   ├── summary_all_loso.csv            # 全6种LOSO方法汇总
    │   ├── loso_comparison_full.png        # 6方法柱状对比图
    │   └── loso_da_delta_full.png          # 逐折提升折线图（CORAL/MMD/DANN vs MLP）
    │
    └── cross_session/                      # 跨session实验结果
        ├── subject_dependent_svm_cross_session_results.npy
        ├── subject_dependent_mlp_cross_session_results.npy
        ├── subject_dependent_mlp_cross_session_coral_results.npy
        ├── subject_dependent_mlp_cross_session_adabn_results.npy
        ├── subject_dependent_mlp_cross_session_adabn_only_results.npy
        ├── summary_full.csv                # 全5种方法汇总（基线+跨session）
        ├── summary_with_coral.csv          # 含CORAL方法汇总
        ├── summary_with_adabn.csv          # 含AdaBN变体汇总
        ├── subject_dependent_full.png      # 5方法柱状对比图
        ├── subject_dependent_delta.png     # 逐折delta图（数据效应 vs 方法效应）
        ├── subject_dependent_with_coral.png   # 6方法柱状图（含CORAL）
        ├── subject_dependent_coral_delta.png  # CORAL贡献逐折delta图
        ├── subject_dependent_with_adabn.png   # 7方法柱状图（含AdaBN变体）
        └── subject_dependent_adabn_delta.png  # AdaBN免费增益逐折delta图
```

---

## 环境依赖

```bash
pip install -r requirements.txt
```

依赖：`numpy`, `scipy`, `scikit-learn`, `torch`, `matplotlib`

---

## 运行说明

### 第一步：基线实验（方向一，注意力机制）

```bash
# 训练 6 个基线模型（Subject-Dependent + LOSO 各3种）
python src/train.py --model svm      --split subject_dependent
python src/train.py --model svm      --split loso
python src/train.py --model mlp      --split subject_dependent
python src/train.py --model mlp      --split loso
python src/train.py --model attn_mlp --split subject_dependent
python src/train.py --model attn_mlp --split loso

# 查看结果表 + 保存 CSV
python src/evaluate.py --save

# 生成注意力权重可视化图
python src/visualize.py
```

结果保存至 `results/`。

---

### 第二步：被试依赖改进——跨 session 迁移学习

```bash
# SVM：合并同被试其他 session 数据（数据效应）
python src/cross_session/train_svm_cross_session.py

# MLP：其他 session 预训练 + 目标 session 微调（方法效应）
python src/cross_session/train_cross_session.py

# 全部 5 种方法对比表
python src/cross_session/evaluate_full.py --save

# 图表（柱状图 + 数据效应 vs 方法效应 delta 图）
python src/cross_session/visualize_full.py
```

结果保存至 `results/cross_session/`。

---

### 第二步补充：被试依赖改进——Session 间域适应

在跨 session 预训练基础上，显式对齐 session 间的特征分布漂移。

#### 2-A CORAL（预训练阶段加协方差对齐）

```bash
# 训练（默认 λ=1.0，可调整）
python src/cross_session/train_cross_session_coral.py
python src/cross_session/train_cross_session_coral.py --lambda_coral 0.5

# 对比表（含 CORAL 方法）
python src/cross_session/evaluate_cross_session_coral.py --save --fold

# 可视化
python src/cross_session/visualize_cross_session_coral.py
```

#### 2-B AdaBN（预训练后无监督替换 BN 统计量）

```bash
# 完整流程：预训练 → AdaBN → 微调
python src/cross_session/train_cross_session_adabn.py

# 消融：仅 AdaBN，不微调（测量零标签自适应增益）
python src/cross_session/train_cross_session_adabn.py --no_finetune

# 对比表（含 AdaBN 变体）
python src/cross_session/evaluate_cross_session_adabn.py --save --fold

# 可视化
python src/cross_session/visualize_cross_session_adabn.py
```

结果保存至 `results/cross_session/`。

---

### 第三步：LOSO 改进——跨被试域适应

#### 3-1 CORAL（相关对齐）

```bash
python src/domain_adaptation/train_coral.py
python src/domain_adaptation/evaluate_coral.py --save
python src/domain_adaptation/visualize_coral.py
```

#### 3-2 MMD（最大均值差异，多核 RBF）

```bash
python src/domain_adaptation/train_mmd.py
python src/domain_adaptation/evaluate_mmd.py --save
python src/domain_adaptation/visualize_mmd.py
```

#### 3-3 DANN（对抗域适应，梯度反转层）

```bash
python src/domain_adaptation/train_dann.py
python src/domain_adaptation/evaluate_dann.py --save
python src/domain_adaptation/visualize_dann.py   # 生成含全部6种方法的综合图
```

结果保存至 `results/domain_adaptation/`。

#### λ 超参数调整（可选）

```bash
python src/domain_adaptation/train_coral.py --lambda_coral 0.5
python src/domain_adaptation/train_mmd.py   --lambda_mmd   2.0
python src/domain_adaptation/train_dann.py  --lambda_dann  0.5
```

---

## 数据划分策略详解

| 策略 | 训练集 | 测试集 | 折数 | 每折训练样本量（约） |
|------|--------|--------|:----:|:-------------------:|
| Subject-Dependent | 同(被试,session)的前9个试次 | 后6个试次 | 36 | ~1600 |
| LOSO | 另外11个被试的全量数据 | 1个被试全量数据 | 12 | ~27000 |

---

## 模型与方法说明

### 基线模型

| 模型 | 输入 | 结构 | 参数量 |
|------|------|------|:------:|
| SVM | (N, 310) | RBF核，C=1.0 | — |
| MLP | (N, 310) | 310→256→128→3，BN+Dropout | ~114K |
| MLP+Attention | (N, 5, 62) | SE频段注意力→SE通道注意力→MLP | ~116K |

注意力机制采用 **SE-Net（Squeeze-and-Excitation）** 风格，非降维设计：输入输出均保持 (N, 5, 62)，展平后接与 MLP 相同的分类头，确保对比公平。使用 Sigmoid 而非 Softmax，允许各频段/通道独立加权。

### 域适应方法（LOSO）

| 方法 | 核心机制 | 额外组件 |
|------|---------|---------|
| CORAL | 对齐源/目标域特征的协方差矩阵 | 一个 loss 项 |
| MMD | 多核 RBF 衡量分布差异（σ=0.5,1,2,5,10） | 一个 loss 项 |
| DANN | 梯度反转层 + 域判别器对抗训练，α 从 0→1 调度 | 域分类器 + GRL |

所有方法训练时使用目标被试的**无标签**测试数据进行分布对齐（标签不泄露）。

### 跨 session 方法（Subject-Dependent）

| 方法 | 核心机制 |
|------|---------|
| SVM + 跨session | 将同被试其他2个session的数据合并进训练集，单次 fit |
| MLP + 跨session预训练 | Phase 1：lr=1e-3 在其他2个session预训练；Phase 2：lr=2e-4 在目标session微调 |
| MLP + CS + CORAL | 预训练阶段加 CORAL loss，对齐 source session 与 target session（无标签）的特征协方差 |
| MLP + CS + AdaBN | 预训练后，用目标 session 全量数据（无标签）重置 BN 的 running mean/var；可选接微调 |

---

## 训练配置

| 参数 | 基线 / 域适应 | 跨session（预训练） | 跨session（微调） |
|------|:-----------:|:------------------:|:----------------:|
| 最大轮数 | 200 | 150 | 80 |
| Early stopping 耐心 | 20 | 20 | 15 |
| 学习率 | 1e-3 | 1e-3 | 2e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 128 | 128 | 128 |
| 验证集比例 | 20% | 20% | 20% |
| 随机种子 | 42 | 42 | 42 |

---

## 引用

> Wei-Long Zheng, and Bao-Liang Lu, "Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks," IEEE Transactions on Autonomous Mental Development 7(3): 162-175, 2015.

> Hu, J., Shen, L., & Sun, G., "Squeeze-and-Excitation Networks," CVPR, 2018.

> Sun, B., & Saenko, K., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation," ECCV Workshop, 2016.

> Gretton, A., et al., "A Kernel Two-Sample Test," JMLR, 2012.

> Long, M., et al., "Learning Transferable Features with Deep Adaptation Networks," ICML, 2015.

> Ganin, Y., et al., "Domain-Adversarial Training of Neural Networks," JMLR, 2016.

> Li, Y., et al., "Revisiting Batch Normalization For Practical Domain Adaptation," ICLR Workshop, 2017.
