# 平滑顺序生成模型代码报告 (Smooth Sequential Model Code Report)

本文档详细分析了 `main_smooth_sequential.py` 的代码逻辑、运行流程及核心算法原理。该版本代码旨在解决标准 MCMC 模型中可能出现的“票数突变”问题，通过引入时间平滑约束，生成更符合真实世界规律（观众偏好具有惯性）的投票估算序列。

---

## 1. 核心设计理念 (Core Philosophy)

### 1.1 问题背景
在标准的逐周 MCMC 估算中，每一周的采样是相互独立的。这可能导致同一个选手在表现相似的连续两周内，估算的粉丝票数出现剧烈波动（例如第3周 20%，第4周突然变成 5%），这不符合真实世界的**观众粘性 (Fan Inertia)** 规律。

### 1.2 解决方案：思路 i (顺序生成平滑)
本代码实现了一种**贪心顺序优化 (Greedy Sequential Optimization)** 策略：
*   **硬约束 (Hard Constraint)**：必须 100% 满足当周的淘汰结果（Consistency）。
*   **软约束 (Soft Constraint)**：在满足硬约束的所有可行解中，选择与上一周**变化最小 (Smoothest)** 的那个解。

---

## 2. 代码执行流程 (Execution Flow)

```mermaid
graph TD
    A[启动 main()] --> B[加载数据 DWTSDataLoader]
    B --> C[初始化 MCMCSampler]
    C --> D[遍历所有赛季 Season 1-34]
    
    subgraph Single Season Processing (process_season_smooth)
    D --> E[初始化上一周状态 Prev_Votes = None]
    E --> F[遍历该赛季每一周 Week 1..T]
    F --> G{是淘汰周?}
    G -- No --> F
    G -- Yes --> H[运行 MCMC 采样]
    H --> I[过滤出 Valid Set (满足一致性的样本)]
    
    I --> J{这是第一周?}
    J -- Yes --> K[选择 Valid Set 中最接近均值的样本]
    J -- No --> L[计算 Valid Set 中每个样本与 Prev_Votes 的距离]
    L --> M[选择距离最小的样本作为本周结果]
    
    M --> N[更新 Prev_Votes]
    K --> N
    N --> F
    end
    
    F --> O[汇总统计与保存结果]
```

### 关键步骤解析

#### 1. 数据准备
*   使用 `DWTSDataLoader` 加载清洗后的数据。
*   识别每个赛季的计分规则 (Rank / Percentage)。

#### 2. MCMC 采样 (探索可行域)
*   对于每一周，首先运行标准的 `MCMCSampler`。
*   **目的**：不是为了直接求期望，而是为了尽可能多地探索**可行解空间**。
*   **过滤**：从采样的 10,000 个样本中，筛选出那些**能正确导致该周淘汰结果**的样本，构成 `valid_samples` 集合。

#### 3. 平滑选择 (Selection Strategy)
函数 `select_smoothest_sample` 是核心：
*   **第一周 (Start)**：没有历史包袱。选择 `valid_samples` 的**几何中心 (Centroid)**，即最接近平均分布的解，假设初始状态没有极端的偏好。
*   **后续周 (Updates)**：
    *   输入：上一周的投票分布 $\mathbf{F}_{t-1}$，当前周的可行解集合 $\{\mathbf{F}^{(k)}_{t}\}$。
    *   计算距离：只比较两周都**共同存活**的选手。
    *   目标：$\min_k \text{Distance}(\mathbf{F}_{t-1}, \mathbf{F}^{(k)}_{t})$。
    *   输出：选择变动最小的那个 $\mathbf{F}^{(best)}_{t}$ 作为本周的估算结果。

#### 4. 兜底机制 (Fallback)
如果 MCMC 采样未能找到任何可行解（极少数情况），代码会退化为 `Rejection Sampling`（拒绝采样）甚至随机猜测，并标记为 `failed`。

---

## 3. 算法原理与数学模型 (Algorithm & Math)

### 3.1 一致性检查 (Consistency Check)
函数 `check_consistency` 用于验证一个样本是否合法。
$$ \text{Valid}(\mathbf{F}) \iff \text{Loser}(\text{Rules}(\mathbf{J}, \mathbf{F})) == \text{Actual Loser} $$
这保证了我们的估算结果在逻辑上是**完全自洽**的。

### 3.2 平滑度量 (Smoothness Metric)
函数 `calculate_smoothness_distance` 定义了时间维度的距离。
假设 $S_t$ 是第 $t$ 周的幸存者集合，$S_{t-1}$ 是第 $t-1$ 周的幸存者集合。共同集 $C = S_t \cap S_{t-1}$。
对于任意一个候选投票向量 $\mathbf{V}$，其与历史向量 $\mathbf{V}_{prev}$ 的距离 $D$ 定义为：

$$ \mathbf{v}' = \text{Normalize}(\{v_i | i \in C\}) $$
$$ \mathbf{v}_{prev}' = \text{Normalize}(\{v_{prev, i} | i \in C\}) $$
$$ D = \sqrt{ \sum_{i \in C} (v'_i - v'_{prev, i})^2 } $$

**注意**：
1.  必须只比较共同选手。
2.  必须重新归一化（Normalize），因为两周的总人数不同，直接比较数值没有意义（少了一个人，其他人的比例自然会上升）。

---

## 4. 代码亮点 (Key Features)

1.  **动态维护状态**：变量 `prev_votes` 和 `prev_names` 在循环中不断更新，像接力棒一样传递历史信息。
2.  **鲁棒的距离计算**：处理了选手名单变化（有人淘汰、有人加入）的情况，只在交集上计算距离。
3.  **混合策略**：结合了 MCMC（高效探索空间）和贪心搜索（优化时间连贯性），比单纯的平滑先验 MCMC 更容易实现且计算成本更低。

---

## 5. 局限性与改进 (Limitations)

*   **贪心算法的近视性**：当前的选择只考虑了与上一周最接近，可能会导致“误差累积”，使后续周次的可行解空间越来越窄。
*   **第一周的假设**：假设第一周是“平均”的，这可能低估了某些自带流量大咖的初始票数。

该代码非常适合作为论文中 **"Model Refinement" (模型改进)** 或 **"Alternative Approach"** 的内容，展示我们不仅考虑了静态的合理性，还考虑了动态的合理性。
