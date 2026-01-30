# DWTS 粉丝投票反演系统 (MCM 2026 C题)

基于贝叶斯逆统计学 (Bayesian Inverse Statistics) 的《与星共舞》隐含投票数据重构系统。

## 🚀 项目概览 (Project Overview)

本项目专为 **2026 MCM Problem C** 设计，旨在解决从未观测的粉丝投票数据中反推淘汰结果这一“不适定逆问题” (Ill-posed Inverse Problem)。

该系统采用 **贝叶斯马尔可夫链蒙特卡洛 (MCMC)** 框架：
1.  **重构历史**: 估算 34 个赛季中每一周、每位选手的粉丝投票概率分布。
2.  **不确定性量化**: 用最高后验密度区间 (HPDI) 替代单一的点估计，以此来衡量我们对估算结果的“确信程度”。
3.  **反事实推演**: 允许进行“如果...会怎样” (What-If) 分析，测试争议结果（如 Bobby Bones 夺冠）在不同规则下是否会改变。

## 📂 项目结构 (Project Structure)

```bash
.
├── 2026_MCM_Problem_C_Data.csv      # 美赛官方提供的原始数据集
├── engineered_data.csv              # 经过特征工程处理的数据集 (自动生成)
├── Data_Dictionary.md               # 新增特征的详细定义文档
├── main.py                          # 单赛季估算模型的主程序入口
├── main_hierarchical.py             # 高级层次贝叶斯模型的程序入口
├── analyze_special_cases.py         # 用于提取历史淘汰真值的脚本
├── src/                             # 核心源代码库
│   ├── data_loader.py               # ETL 数据管道与博弈状态重构
│   ├── mcmc_sampler.py              # Metropolis-Hastings MCMC 采样引擎
│   ├── hierarchical_sampler.py      # 层次贝叶斯模型扩展
│   └── diagnostics.py               # 收敛性检测工具 (R-hat, ESS)
├── results/                         # 结果输出目录
│   ├── estimation_results.pkl       # 序列化保存的 MCMC 链数据
│   └── season_X_visualizations/     # 自动生成的用于论文的图表
└── README.md                        # 本说明文件
```

## 🧠 核心方法论 (Core Methodology)

### 1. 贝叶斯逆建模 (Bayesian Inverse Modeling)
我们将未知的粉丝得票 $F$ 视为潜变量 (Latent Variables)。
*   **似然函数 $P(E|F, J)$**: 在给定我们的猜测票数 $F$ 和已知评委分 $J$ 的情况下，观察到真实淘汰结果 $E$ 的概率。如果某个猜测导致了错误的淘汰者，其概率会骤降至 0。
*   **先验分布 $P(F)$**: 使用 `Dirichlet(α)` 先验来建模 $N$ 名选手之间的选票分布，确保 $\sum F_i = 1$。

### 2. MCMC 约束满足 (Constraint Satisfaction)
为了处理“所有得票占比之和必须为 100%”的单纯形约束，我们在采样的过程中使用了 **Logit 空间** 进行无约束漫步，并通过 Softmax 函数映射回单纯形空间。
*   **采样器**: 自适应 Metropolis-Hastings 算法，带有多元高斯提议分布。
*   **规则引擎**: 根据赛季自动切换淘汰逻辑：
    *   **Rank Method** (S1-2, S28+): 评委排名 + 粉丝排名之和。
    *   **Percent Method** (S3-27): 评委得分占比 + 粉丝投票占比。
    *   **Judges' Save**: 针对“倒数前两名对决”的特殊逻辑。

### 3. 特征工程 (Feature Engineering)
我们对原始数据进行了扩展，增加了 100+ 个新特征 (详见 `Data_Dictionary.md`)，包括：
*   **`weekX_judge_rank`**: 标准化的竞赛排名。
*   **`total_fan_saves_bottom1`**: 这是一个极高信噪比的“粉丝基数”代理变量——统计选手在评委打分倒数第一的情况下幸存的次数。

## 🛠️ 使用指南 (Usage)

###前置条件
*   Python 3.10+
*   依赖库: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

### 快速启动
1.  **数据预处理**: 系统会自动检查并生成 `engineered_data.csv`。
2.  **运行估算**:
    ```bash
    python main.py
    ```
    这将针对默认赛季（如 Season 1）运行标准 MCMC 模型并生成诊断图表。

3.  **层次化模型分析** (高级):
    ```bash
    python main_hierarchical.py
    ```
    运行层次模型，汇聚跨周/跨赛季的信息，以更鲁棒地估算“名人热度”参数。

## 📊 输出与诊断 (Outputs & Diagnostics)

系统会生成：
*   **后验分布图**: 直方图展示每位明星可能的得票范围。
*   **迹图 (Trace Plots)**: 用于视觉检查 MCMC 的混合情况。
*   **收敛性指标**:
    *   **R-hat (Gelman-Rubin)**: 值 < 1.1 表示链已收敛。
    *   **接受率 (Acceptance Rate)**: 监控采样效率 (目标约为 23%)。

## 📝 许可证 (License)
MIT License. 仅供教育与竞赛目的使用。

