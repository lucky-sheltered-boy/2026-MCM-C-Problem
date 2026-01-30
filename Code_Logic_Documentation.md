# 代码逻辑与运行流程文档 (Code Logic & Execution Flow Documentation)

本文档详细解析了代码库的实现细节，阐述了从原始数据摄入到贝叶斯推断结果输出的端到端执行流程。它可作为本项目的技术白皮书。

---

## 1. 系统架构图 (System Architecture Diagram)
```mermaid
graph TD
    A[Raw CSV Data(原始数据)] -->|Pre-processing| B[Engineered Data(特征工程数据)]
    B -->|src.data_loader| C[Season Game State(赛季博弈状态)]
    C -->|Input State| D[MCMC Sampler Engine(MCMC采样引擎)]
    
    subgraph Bayesian Inference Core(贝叶斯推断核心)
    D -->|Proposal (Logit Space)| E{Metropolis-Hastings Step}
    E -->|Likelihood Function| F[Rule Engine(规则引擎: Rank/Percent)]
    F -->|Prior (Dirichlet)| G[Accept/Reject(接受/拒绝)]
    G -->|Update Chain| D
    end
    
    D -->|Posterior Samples| H[Diagnostics(诊断: R-hat)]
    H -->|Analysis| I[Output Visualizations(输出可视化)]
```

---

## 2. 详细执行流程 (Execution Flow Step-by-Step)

### 第一阶段：数据初始化 (ETL)
**对应文件**: `main.py` -> `src/data_loader.py`

1.  **数据摄入 (Ingestion)**：系统读取 `engineered_data.csv`。该文件已包含诸如 `weekX_judge_rank`（评委排名）和 `weekX_percent_score`（评委得分率）等计算好的指标。
2.  **状态重构 (State Reconstruction)**：针对特定目标赛季（如第1季），`DataLoader` 会重构每一周的“博弈状态 (Game State)”：
    *   **活跃集合 (Active Set)**：本周谁还在场上？
    *   **淘汰目标 (Elimination Target)**：本周实际上谁被淘汰了？（解析 `results` 列）。
    *   **评委输入 (Judge Inputs)**：本周评委的具体打分或排名是多少？
3.  **输出**：一个结构化的 `season_data` 字典，包含每一周的“地面真值 (Ground Truth)”。

### 第二阶段：贝叶斯逆向采样 (Bayesian Inverse Sampling)
**对应文件**: `src/mcmc_sampler.py`

核心目标是估算每一周 $w$ 的未知向量 $\mathbf{F}_w = [f_1, f_2, ..., f_n]$（即 $n$ 位选手的粉丝得票率）。

#### 步骤 2.1：空间变换 (Space Transformation)
由于粉丝投票之和必须为 1（即 $\sum f_i = 1$），直接在这个“单纯形 (Simplex)”上采样非常困难。
*   **技术手段**：我们在无约束的 **Logit 空间** ($\mathbf{X} \in \mathbb{R}^n$) 中进行采样。
*   **数学变换**：$\mathbf{F} = \text{Softmax}(\mathbf{X})$。
*   **优势**：我们可以自由使用高斯随机游走（Gaussian Random Walks）来生成提议，完全不用担心边界约束（如不能小于0或大于1）。

#### 步骤 2.2：似然函数构建 ($P(E|F)$)
给定一个猜测的粉丝投票分布 $\mathbf{F}$，观察到真实淘汰结果 $E$ 的概率有多大？
*   **情形 A：排名法 (Rank Method)** (S1-2, S28+)
    *   计算 `总排名 (Total Rank) = 评委排名 + 粉丝排名(由F计算)`。
    *   找出模拟的输家（总排名数值最大者）。
    *   与*真实*输家对比。
    *   如果 模拟输家 == 真实输家，则似然度 = 高。否则，似然度 $\approx 0$。
*   **情形 B：百分比法 (Percent Method)** (S3-27)
    *   计算 `总分 (Total Score) = 评委得分率 + 粉丝得票率(F)`。
    *   找出模拟的输家（总分最低者）。
    *   与真实输家对比。

#### 步骤 2.3：Metropolis-Hastings 循环
进行 `n_iterations` 次（例如 10,000 次）迭代：
1.  **提议 (Propose)**：微扰当前状态：$\mathbf{X}_{new} = \mathbf{X}_{old} + \mathcal{N}(0, \sigma)$。
2.  **评估 (Evaluate)**：计算接受率 $\alpha = \frac{P(E|F_{new})P(F_{new})}{P(E|F_{old})P(F_{old})}$。
3.  **决策**：根据 $\alpha$ 决定是接受新状态还是保留旧状态，以维持详细平衡 (Detailed Balance)。

### 第三阶段：诊断与验证 (Diagnostics & Validation)
**对应文件**: `src/diagnostics.py`

*   **多链验证 (Multi-Chain Verification)**：我们并行运行 3 条从随机位置出发的 MCMC 链。
*   **Gelman-Rubin 统计量 ($\hat{R}$)**：
    *   衡量链*间*方差与链*内*方差的比值。
    *   如果 $\hat{R} < 1.1$，意味着收敛（即所有链都找到了同一个解）。
*   **一致性得分 (Consistency Score)**：衡量有多少百分比的后验样本能够成功复现历史淘汰结果。

---

## 3. 关键算法与公式 (Key Algorithms & Formulas)

### 3.1 Softmax 变换
将无约束的提议模拟值 $\mathbf{x}$ 映射为合法的投票百分比 $\mathbf{p}$：
$$ p_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} $$

### 3.2 后验概率公式
$$ P(\text{Votes} | \text{Result}) \propto \underbrace{P(\text{Result} | \text{Votes})}_{\text{Likelihood}} \times \underbrace{P(\text{Votes})}_{\text{Prior}} $$
*   **先验 (Prior)**：$\text{Dirichlet}(\alpha=1.5)$（弱信息先验，假设票数分布相对均匀，但也允许出现人气高峰）。

---

## 4. 输出解读 (Output Interpretation)

最终输出**不是**一个单一数值（例如“Jerry 获得了 20%”），而是一个**概率分布**：
*   **均值 (Mean)**：最可能的得票份额。
*   **标准差 (Standard Deviation)**：不确定性。高标准差意味着“无论他得多少票，结果都不会变（比如评委分太高了，怎么投都安全）”，因此模型对具体票数不敏感。
*   **HPDI (95%)**：我们有 95% 的把握认为真实的得票率落在这个区间内。

这种概率输出完美回应了 Problem C 关于“不确定性度量 (measure of certainty)”的要求。
