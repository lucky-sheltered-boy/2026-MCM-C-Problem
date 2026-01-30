# DWTS 粉丝投票估算系统

基于贝叶斯MCMC方法的《与星共舞》粉丝投票逆向估算系统

## 项目简介

本项目针对2026 MCM C题，使用贝叶斯马尔可夫链蒙特卡洛(Bayesian MCMC)方法估算《与星共舞》节目中未公开的粉丝投票数据。系统能够：

- ✅ 估算每周每位参赛者的粉丝投票份额
- ✅ 提供估算结果的不确定性度量（HPDI、后验标准差等）
- ✅ 验证模型与实际淘汰结果的一致性
- ✅ 支持三种不同的投票规则（排名法、百分比法、Bottom Two机制）
- ✅ 自动诊断MCMC收敛性（Gelman-Rubin统计量）

## 项目结构

```
.
├── main.py                      # 主程序入口
├── src/
│   ├── data_loader.py          # 数据加载模块（使用预处理后的数据）
│   ├── mcmc_sampler.py         # MCMC采样器
│   └── diagnostics.py          # 模型诊断与验证
├── 2026-MCM-C-Problem/         # 队友预处理数据（git clone）
│   └── engineered_data.csv     # 预处理后的特征数据
├── results/                     # 输出结果目录
│   ├── estimation_results.pkl  # 估算结果
│   ├── results_summary.json    # 结果摘要
│   └── season_X_visualizations/ # 各赛季可视化
├── test_modules.py             # 模块测试脚本
└── README.md                    # 本文件
```

## 环境配置

### 创建conda环境
```bash
conda create -n MCM python=3.11 -y
conda activate MCM
pip install numpy pandas scipy matplotlib seaborn
```

## 使用方法

### 1. 准备数据

确保 `2026-MCM-C-Problem/engineered_data.csv` 存在（已由队友预处理）。

### 2. 运行完整分析

```bash
conda activate MCM
python main.py
```

这将执行：
- 数据预处理
- MCMC采样估算
- 结果可视化

### 3. 自定义分析

```python
from main import DWTSVotingEstimator

# 初始化
estimator = DWTSVotingEstimator("2026_MCM_Problem_C_Data.csv")

# 处理指定赛季
estimator.load_and_process_data(seasons=[1, 2, 3])

# 估算单个赛季
estimator.estimate_season(season=1, n_chains=3)

# 可视化结果
estimator.visualize_season(season=1)
```

## 核心算法

### Metropolis-Hastings采样

1. **状态空间**: 使用Softmax变换确保粉丝投票份额和为1
2. **似然函数**: 根据赛季规则动态选择（排名法/百分比法/Bottom Two）
3. **先验分布**: Dirichlet(α=1.5) 先验
4. **提议分布**: 在无约束空间添加高斯噪声

### 收敛性诊断

- **Gelman-Rubin统计量**: R-hat < 1.1 表示收敛
- **有效样本量**: ESS评估独立样本数量
- **后验预测检查**: 验证模型一致性

### 不确定性度量

- **95% HPDI**: 最高后验密度区间
- **后验标准差**: 估算的变异性
- **变异系数**: 标准化的不确定性度量

## 输出说明

### 1. 估算结果 (estimation_results.pkl)

每周的估算结果包含：
- `fan_votes_mean`: 粉丝投票份额的后验均值
- `fan_votes_std`: 后验标准差
- `fan_votes_hpdi`: 95% HPDI区间
- `consistency_score`: 模型一致性分数
- `r_hat`: Gelman-Rubin统计量
- `converged`: 是否收敛

### 2. 可视化图表

- **consistency_trend.png**: 各周一致性趋势
- **certainty_metrics.png**: 估算不确定性分析
- **posterior_week_X.png**: 后验分布直方图

## 方法论亮点

1. **单纯形约束处理**: Softmax变换优雅地处理了概率和为1的约束
2. **赛季规则适配**: 自动识别并应用不同赛季的投票规则
3. **多链验证**: 运行多条MCMC链确保结果可靠性
4. **完整诊断**: 提供全面的收敛性和不确定性分析

## 技术特点

- ✨ 模块化设计，易于扩展
- ✨ 完整的错误处理和日志输出
- ✨ 支持大规模数据（34赛季）处理
- ✨ 自动生成专业级可视化
- ✨ 结果可序列化存储和加载

## 后续扩展

可以在此基础上添加：
- PyMC3/Stan集成（使用NUTS采样器加速）
- 交互式可视化（Plotly）
- 因素分析模型（名人特征、舞伴影响等）
- 投票方法比较分析

## 理论基础

详细的数学推导和算法设计请参考：
`Bayesian MCMC Modeling for Dancing with the Stars Voting Analysis.md`

## 联系方式

如有问题或建议，欢迎提出Issue。

## 许可证

MIT License
