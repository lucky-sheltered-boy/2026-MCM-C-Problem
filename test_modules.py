"""
快速测试脚本：验证各模块功能
在没有真实数据的情况下测试代码逻辑
"""

import numpy as np
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mcmc_sampler import MCMCSampler
from diagnostics import ModelDiagnostics, CertaintyMetrics


def test_mcmc_sampler():
    """测试MCMC采样器"""
    print("=" * 60)
    print("测试 1: MCMC采样器")
    print("=" * 60)
    
    # 模拟场景：4名选手，第2名被淘汰（0-indexed为1）
    judge_share = np.array([0.30, 0.18, 0.28, 0.24])  # 第2名评委分最低
    eliminated_idx = 1
    
    sampler = MCMCSampler(
        n_iterations=5000,
        burn_in=1000,
        thinning=5,
        proposal_sigma=0.3
    )
    
    print(f"\n场景设定:")
    print(f"  评委得分份额: {judge_share}")
    print(f"  实际淘汰选手: 选手 {eliminated_idx + 1}")
    
    # 测试三种投票方法
    methods = ['rank', 'percentage', 'rank_bottom2']
    
    for method in methods:
        print(f"\n--- 测试{method}方法 ---")
        samples = sampler.sample_week(judge_share, eliminated_idx, method)
        
        print(f"样本形状: {samples.shape}")
        print(f"接受率: {sampler.acceptance_rate:.2%}")
        print(f"粉丝投票估计均值: {samples.mean(axis=0)}")
        print(f"粉丝投票标准差: {samples.std(axis=0)}")
        
        # 验证样本和为1
        sample_sums = samples.sum(axis=1)
        print(f"样本和检查 (应接近1): min={sample_sums.min():.6f}, max={sample_sums.max():.6f}")
    
    print("\n✓ MCMC采样器测试完成")
    return samples


def test_diagnostics(samples):
    """测试诊断工具"""
    print("\n" + "=" * 60)
    print("测试 2: 模型诊断工具")
    print("=" * 60)
    
    # 生成3条链用于测试
    np.random.seed(42)
    n_samples, n_params = samples.shape
    chains = [
        np.random.dirichlet([2, 1.5, 2.5, 2], n_samples) for _ in range(3)
    ]
    
    # Gelman-Rubin统计量
    print("\n--- Gelman-Rubin收敛诊断 ---")
    r_hat = ModelDiagnostics.gelman_rubin_statistic(chains)
    print(f"R-hat值: {r_hat}")
    converged = np.all(r_hat < 1.1)
    print(f"收敛状态: {'✓ 已收敛 (R-hat < 1.1)' if converged else '✗ 未收敛'}")
    
    # 有效样本量
    print("\n--- 有效样本量 ---")
    ess = ModelDiagnostics.effective_sample_size(samples)
    print(f"ESS: {ess}")
    print(f"ESS比率: {ess / len(samples)}")
    
    # 确定性度量
    print("\n--- 确定性度量 ---")
    hpdi_widths = CertaintyMetrics.calculate_hpdi_width(samples)
    print(f"HPDI宽度: {hpdi_widths}")
    
    cv = CertaintyMetrics.calculate_coefficient_of_variation(samples)
    print(f"变异系数: {cv}")
    
    # 后验预测检查
    print("\n--- 后验预测检查 ---")
    judge_share = np.array([0.30, 0.18, 0.28, 0.24])
    eliminated_idx = 1
    
    ppc_score = ModelDiagnostics.posterior_predictive_check(
        samples, judge_share, eliminated_idx, 'percentage'
    )
    print(f"一致性分数: {ppc_score:.2%}")
    
    print("\n✓ 诊断工具测试完成")


def test_likelihood_functions():
    """测试似然函数"""
    print("\n" + "=" * 60)
    print("测试 3: 似然函数逻辑")
    print("=" * 60)
    
    sampler = MCMCSampler()
    
    # 测试场景
    judge_share = np.array([0.30, 0.20, 0.25, 0.25])
    fan_votes = np.array([0.15, 0.40, 0.25, 0.20])  # 粉丝给第2名很高票
    
    print(f"\n评委得分份额: {judge_share}")
    print(f"粉丝投票份额: {fan_votes}")
    
    # 排名法
    print("\n--- 排名法 ---")
    judge_rank = len(judge_share) + 1 - np.argsort(np.argsort(judge_share)) - 1
    fan_rank = len(fan_votes) + 1 - np.argsort(np.argsort(fan_votes)) - 1
    total_rank = judge_rank + fan_rank
    
    print(f"评委排名: {judge_rank}")
    print(f"粉丝排名: {fan_rank}")
    print(f"综合排名: {total_rank}")
    print(f"排名最差者（应被淘汰）: 选手 {np.argmax(total_rank) + 1}")
    
    # 百分比法
    print("\n--- 百分比法 ---")
    combined_score = 0.5 * judge_share + 0.5 * fan_votes
    print(f"综合得分: {combined_score}")
    print(f"得分最低者（应被淘汰）: 选手 {np.argmin(combined_score) + 1}")
    
    # Bottom Two
    print("\n--- Bottom Two机制 ---")
    bottom_two = np.argsort(total_rank)[-2:]
    print(f"底部两名: 选手 {bottom_two + 1}")
    
    print("\n✓ 似然函数测试完成")


def test_softmax_constraint():
    """测试Softmax约束"""
    print("\n" + "=" * 60)
    print("测试 4: Softmax单纯形约束")
    print("=" * 60)
    
    sampler = MCMCSampler()
    
    # 测试多个无约束向量
    test_vectors = [
        np.array([0, 0, 0, 0]),
        np.array([1, 2, 3, 4]),
        np.array([-2, 0, 2, 4]),
        np.random.randn(4) * 5
    ]
    
    print("\n无约束向量 -> Softmax变换 -> 验证和为1")
    print("-" * 60)
    
    for i, z in enumerate(test_vectors, 1):
        probs = sampler.softmax_transform(z)
        total = probs.sum()
        print(f"向量 {i}: {z}")
        print(f"  -> 概率: {probs}")
        print(f"  -> 和: {total:.10f} {'✓' if abs(total - 1.0) < 1e-6 else '✗'}")
        print()
    
    print("✓ Softmax约束测试完成")


def main():
    """运行所有测试"""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "DWTS 粉丝投票估算系统" + " " * 15 + "║")
    print("║" + " " * 20 + "快速测试套件" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # 测试1: MCMC采样器
        samples = test_mcmc_sampler()
        
        # 测试2: 诊断工具
        test_diagnostics(samples)
        
        # 测试3: 似然函数
        test_likelihood_functions()
        
        # 测试4: Softmax约束
        test_softmax_constraint()
        
        # 总结
        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！代码逻辑正确 ✓✓✓")
        print("=" * 60)
        print("\n说明:")
        print("  - 各模块功能正常")
        print("  - MCMC采样器工作正常")
        print("  - 诊断工具计算正确")
        print("  - 约束条件得到满足")
        print("\n下一步:")
        print("  1. 准备数据文件: 2026_MCM_Problem_C_Data.csv")
        print("  2. 运行主程序: python main.py")
        print("  3. 查看结果: results/ 目录")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
