# MCM 2026 Problem C: DWTS 粉丝投票反演与分析系统

本项目是针对 **2026美赛 (MCM/ICM) C题** 的完整解决方案。我们构建了一个基于 **贝叶斯马尔可夫链蒙特卡洛 (MCMC)** 的逆向统计系统，用于从历史淘汰数据中重构《与星共舞》(Dancing with the Stars) 的隐含粉丝投票数据，并在此基础上进行了深入的反事实分析、敏感性分析及新赛制模拟。

## 🚀 项目亮点 (Highlights)

*   **Q1 历史重构**: 使用 Metropolis-Hastings 算法结合顺序平滑约束，重构了 S1-S33 赛季的每周粉丝得票分布。
*   **Q2 反事实推演**: 模拟了改变评分规则、移除评委等干预措施对历史结果（如 Bobby Bones 夺冠）的影响。
*   **Q3 归因分析**: 量化了舞蹈风格、出场顺序、性别等因素对得分的边际效应。
*   **Q4 新赛制设计**: 提出并模拟了 "Two-Stage Approval" (TSA) 投票系统，旨在提高公平性。

## 📂 项目结构 (Structure)

### 1. 核心模型与数据生成 (Core Model)
*   `main.py` / `main_smooth_sequential.py`: **[Entry Point]** MCMC 采样主程序。生成符合 100% 历史一致性的粉丝投票数据，使用顺序平滑 (Sequential Smoothing) 确保相邻周数据连贯。
*   `src/`: 包含核心算法库 (`mcmc_sampler.py`, `smooth_sampler.py`, `data_loader.py`, `diagnostics.py`)。

### 2. 问题分析脚本 (Analysis Scripts)
*   **Q2 分析**: `analysis_q2_counterfactual.py` - 运行反事实模拟（如采用 S28 规则重演 S27）。
*   **Q3 分析**: `analysis_q3_factors.py` - 使用分层线性模型 (Hierarchical Linear Model) 分析得分影响因子。
*   **Q4 分析**: `analysis_q4_new_system.py` - 模拟新投票系统的公平性与偏见减少效果。
*   **敏感性分析**: `sensitivity_analysis.py` - 测试模型参数（如 $\alpha$）对结果的稳定性影响。

### 3. 可视化 (Visualization)
*   `plot_task1_fancy.py`: 绘制 Q1 粉丝投票热力图与区间图。
*   `plot_q2_fancy.py`: 绘制反事实改变概率图。
*   `plot_q3_fancy.py`: 绘制影响因子系数图。
*   `plot_sensitivity_fancy.py`: 绘制参数敏感性分析图。

### 4. 文档 (Documentation)
*   `docs/`: 包含各问题的详细推导报告。
*   `Code_Logic_Documentation.md`: 代码逻辑与算法原理详解。
*   `Model_Evaluation_and_Discussion.md`: 模型的优缺点评估。
*   `Sequential_Model_Documentation.md`: 顺序平滑模型的数学细节。
*   `Analysis_Q*_Documentation.md`: 针对各问题的详细分析文档。

## 🛠️ 安装与运行 (Usage)

### 环境要求
本项目基于 Python 3.9+。请安装依赖：
```bash
pip install -r requirements.txt
```

### 运行步骤
1.  **生成数据** (这一步大约需要 5-10 分钟，结果存入 `results/`):
    ```bash
    python main.py
    ```
    *注：`main.py` 实际上是运行 sequential smooth sampler，旨在生成平滑且符合历史事实的数据链。*

2.  **运行各问题分析**:
    ```bash
    python analysis_q2_counterfactual.py  # 运行 Q2
    python analysis_q3_factors.py       # 运行 Q3
    python analysis_q4_new_system.py    # 运行 Q4
    ```

3.  **生成绘图**:
    ```bash
    python plot_task1_fancy.py
    python plot_q2_fancy.py
    # ... 其他绘图脚本
    ```

## 📊 数据说明 (Data)

*   `2026_MCM_Problem_C_Data.csv`: 官方原始数据。
*   `engineered_data.csv`: 清洗并添加特征后的中间数据（由程序自动生成）。
*   `results/mcmc_smooth_results.csv`: 核心产出，包含重构的每一周每位选手的得票率分布 (Mean, HPDI)。

## 🧠 方法论简介

我们将未知的粉丝得票视为**潜变量**，利用贝叶斯法则 $P(F|E) \propto P(E|F)P(F)$ 进行推断：
*   **似然 $P(E|F)$**: 只要模拟的得票 $F$ 导致了与历史事实不符的淘汰结果，则概率为 0，否则为 1。
*   **先验 $P(F)$**: 弱信息 Dirichlet 先验，防止过度拟合。
*   **采样**: 使用 Logit 变换后的 Random Walk Metropolis-Hastings 算法，在此基础上增加了时间平滑约束，确保选手的粉丝支持率不会在相邻周发生剧烈突变。

---
*MCM 2026 Campaign Codebase*
