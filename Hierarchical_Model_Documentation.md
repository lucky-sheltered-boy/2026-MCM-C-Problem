# 层次贝叶斯模型代码逻辑文档 (Hierarchical Bayesian Model Logic)

本文档详细解析了 `main_hierarchical.py` 及其依赖模块 `src.hierarchical_sampler.py` 的实现原理。这是本项目的高级模型部分，用于在更宏观的层面捕捉选手的“潜在人气参数”。

---

## 1. 核心思想：为什么需要层次模型？(Why Hierarchical?)

标准模型（在 `main.py` 中）是**独立**地估算每一周的投票。
然而，选手的“人气”具有**跨时间的稳定性**。如果一位选手在第1周得票很高，他在第2周得票大概率也不会太低。

**层次贝叶斯模型 (HBM)** 利用了这种信息共享机制：
*   **层级 1 (Player Level)**：每位选手有一个潜在的“基础人气”参数 $\theta_i$ (Base Popularity)。
*   **层级 2 (Week Level)**：每周的实际得票 $F_{i,t}$ 是由基础人气 $\theta_i$ 加上每周的随机波动产生的。

通过这种方式，即使某些周次的数据很少（比如很早就被淘汰了），模型也可以借用该选手其他表现较好周次的信息，或者借用整个群体的信息来进行更准确的估算。

---

## 2. 系统流程图 (Flowchart)

```mermaid
graph TD
    A[Data Ingestion] -->|Load| B[Engineered Data]
    B -->|Extract| C[Global Contestant Pool]
    
    subgraph Hierarchical Model Structure
    C -->|Params| D[Theta: Base Popularity (N)]
    D -->|Prior| E[Hyper-Prior: Alpha ~ Gamma]
    D -->|Generate| F[Weekly Votes: F_it]
    F -->|Constraint| G[Sum to 1 per week]
    end
    
    subgraph MCMC Sampling
    H[Metropolis-Hastings] -->|Propose Theta| D
    H -->|Propose Weekly Noise| F
    G -->|Likelihood| I[Match Elimination History]
    I -->|Accept/Reject| H
    end
    
    H -->|Posterior| J[Popularity Ranking]
```

---

## 3. 代码执行流程 (Execution Flow)

### 第一阶段：数据池化 (Data Pooling)
**文件**: `main_hierarchical.py` -> `load_and_process_data`

1.  **跨赛季聚合**：代码可以一次性加载多个赛季（例如 S1-S5）。
2.  **ID 映射**：建立 `Global ID` 映射表，因为不同赛季的选手ID需要统一管理。
3.  **稀疏矩阵构建**：由于不是所有选手都参加所有赛季，数据结构是一个稀疏的时间序列矩阵。

### 第二阶段：层次采样器 (Hierarchical Sampler)
**文件**: `src/hierarchical_sampler.py`

#### 步骤 2.1：参数定义
模型不再直接采样票数，而是采样：
1.  **$\theta$ (Theta)**：长度为 $N$ 的向量，代表每个选手的**固有吸引力 (Intrinsic Appeal)**。
2.  **$\sigma_{week}$**：周际波动方差。

#### 步骤 2.2：生成过程 (Generative Process)
每一周的得票 $\mathbf{F}_t$ 生成逻辑如下：
$$ \mathbf{X}_{t} \sim \mathcal{N}(\boldsymbol{\theta}, \sigma_{week}^2 \mathbf{I}) $$
$$ \mathbf{F}_{t} = \text{Softmax}(\mathbf{X}_{t}) $$

这里，$\theta$ 起到了“锚点 (Anchor)”的作用，拉住了每周的波动，使其不至于太过离谱。

#### 步骤 2.3：联合似然函数
现在的目标函数（后验概率）变得更加复杂：
$$ P(\theta, \mathbf{F}_{1...T} | E_{1...T}) \propto \underbrace{\prod_{t=1}^T P(E_t | \mathbf{F}_t)}_{\text{Weekly Data Fit}} \times \underbrace{\prod_{t=1}^T P(\mathbf{F}_t | \theta)}_{\text{Hierarchical Structure}} \times \underbrace{P(\theta)}_{\text{Hyper-Prior}} $$

代码中的 `log_likelihood` 函数并行计算所有周次的淘汰吻合度，并加上 $\theta$ 对 $\mathbf{F}_t$ 的约束惩罚。

### 第三阶段：全剧终推断 (Inference)
**文件**: `main_hierarchical.py` -> `analyze_results`

模型输出的不再仅仅是某一周的得票，而是：
*   **选手历史地位排名**：基于 $\theta$ 的排序，谁是 DWTS 历史上基础人气最高的明星？
*   **异常检测**：某些周次的实际得票 $F_{i,t}$ 远超其基础人气 $\theta_i$？这说明发生了“爆冷”或“刷票”事件。

---

## 4. 关键算法创新点 (Key Innovations)

### 4.1 借力打力 (Borrowing Strength)
这是 HBM 的核心优势。对于那些只在第 2 周就被淘汰的选手，标准模型很难估算他们的真实人气（因为数据点太少）。但 HBM 可以通过群体分布（Hyper-parameters）对他们进行合理的推断，避免过拟合。

### 4.2 鲁棒性 (Robustness)
相比于标准模型，层次模型对“噪声”不那么敏感。如果某一周评委分很低但没淘汰，标准模型可能会认为该选手这周得票率 90%；而层次模型会认为“他平时没那么火，这周可能只是得票率 40% 就够活命了”，给出的估计更保守、更合理。

---

## 总结

`main_hierarchical.py` 是本论文的**高阶武器**。它从“分析一场比赛”上升到了“分析整个联赛机制”的高度，非常适合用于论文后半部分的深入讨论 (In-depth Analysis) 和模型改进 (Model Refinement) 章节。
