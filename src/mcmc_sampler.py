"""
MCMC采样器模块 (MCMC Sampler Module)
实现Metropolis-Hastings算法估算粉丝投票
"""

import numpy as np
from typing import Tuple, Callable, Dict, List
from scipy.special import softmax
import warnings


class MCMCSampler:
    """贝叶斯MCMC采样器用于估算粉丝投票"""
    
    def __init__(self, n_iterations: int = 10000, burn_in: int = 2000, 
                 thinning: int = 5, proposal_sigma: float = 0.3):
        """
        初始化MCMC采样器
        
        Args:
            n_iterations: 总迭代次数
            burn_in: Burn-in期（丢弃前N个样本）
            thinning: 抽稀间隔（每隔N个样本保留1个）
            proposal_sigma: 提议分布的标准差
        """
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thinning = thinning
        self.proposal_sigma = proposal_sigma
        
        # 采样结果存储
        self.samples = None
        self.acceptance_rate = 0.0
        
    def softmax_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax变换：从无约束空间映射到单纯形
        
        Args:
            z: 无约束向量
            
        Returns:
            概率向量（和为1）
        """
        return softmax(z)
    
    def propose_new_state(self, z_current: np.ndarray) -> np.ndarray:
        """
        生成新的候选状态（在无约束空间中添加高斯噪声）
        
        Args:
            z_current: 当前无约束向量
            
        Returns:
            候选无约束向量
        """
        epsilon = np.random.normal(0, self.proposal_sigma, size=z_current.shape)
        z_new = z_current + epsilon
        return z_new
    
    def calculate_log_likelihood_rank(self, fan_votes_share: np.ndarray, 
                                     judge_share: np.ndarray,
                                     eliminated_idx: int,
                                     use_bottom_two: bool = False,
                                     judge_ranks: np.ndarray = None) -> float:
        """
        计算排名法的对数似然
        
        Args:
            fan_votes_share: 粉丝投票份额
            judge_share: 评委得分份额
            eliminated_idx: 实际被淘汰选手的索引
            use_bottom_two: 是否使用Bottom Two机制（28-34赛季）
            judge_ranks: 预计算的评委排名（可选，若提供则直接使用）
            
        Returns:
            对数似然值
        """
        # 使用预计算的评委排名，或自行计算（兼容旧代码）
        if judge_ranks is not None:
            judge_rank = judge_ranks
        else:
            judge_rank = len(judge_share) + 1 - np.argsort(np.argsort(judge_share)) - 1
        
        # 计算粉丝排名（rank越小越好，从1开始）
        fan_rank = len(fan_votes_share) + 1 - np.argsort(np.argsort(fan_votes_share)) - 1
        
        # 综合排名（值越大越差）
        total_rank = judge_rank + fan_rank
        
        if use_bottom_two:
            # 找出排名最差的两位（total_rank最大的两个）
            bottom_two_indices = np.argsort(total_rank)[-2:]
            
            # 似然：实际淘汰者在bottom two中
            if eliminated_idx in bottom_two_indices:
                return 0.0  # log(1) = 0
            else:
                return -np.inf  # log(0) = -inf
        else:
            # 找出排名最差的选手（total_rank最大）
            worst_idx = np.argmax(total_rank)
            
            # 似然：实际淘汰者就是排名最差者
            if worst_idx == eliminated_idx:
                return 0.0  # log(1) = 0
            else:
                return -np.inf  # log(0) = -inf
    
    def calculate_log_likelihood_percentage(self, fan_votes_share: np.ndarray,
                                           judge_share: np.ndarray,
                                           eliminated_idx: int) -> float:
        """
        计算百分比法的对数似然
        
        Args:
            fan_votes_share: 粉丝投票份额
            judge_share: 评委得分份额
            eliminated_idx: 实际被淘汰选手的索引
            
        Returns:
            对数似然值
        """
        # 综合得分（50%评委 + 50%粉丝）
        combined_score = 0.5 * judge_share + 0.5 * fan_votes_share
        
        # 找出得分最低的选手
        worst_idx = np.argmin(combined_score)
        
        # 似然：实际淘汰者就是得分最低者
        if worst_idx == eliminated_idx:
            return 0.0  # log(1) = 0
        else:
            return -np.inf  # log(0) = -inf
    
    def calculate_log_prior(self, fan_votes_share: np.ndarray) -> float:
        """
        计算先验概率（使用Dirichlet先验）
        
        Args:
            fan_votes_share: 粉丝投票份额
            
        Returns:
            对数先验概率
        """
        # 使用对称Dirichlet先验，alpha=1（均匀分布）
        # 或alpha稍大于1，表示略微倾向于均匀分布
        alpha = 1.5
        n = len(fan_votes_share)
        
        # Dirichlet的对数概率密度（忽略归一化常数）
        log_prior = np.sum((alpha - 1) * np.log(fan_votes_share + 1e-10))
        
        return log_prior
    
    def metropolis_hastings_step(self, z_current: np.ndarray,
                                  judge_share: np.ndarray,
                                  eliminated_idx: int,
                                  likelihood_func: Callable) -> Tuple[np.ndarray, bool]:
        """
        执行一步Metropolis-Hastings采样
        
        Args:
            z_current: 当前无约束向量
            judge_share: 评委得分份额
            eliminated_idx: 被淘汰选手索引
            likelihood_func: 似然函数
            
        Returns:
            (新状态, 是否接受)
        """
        # 生成候选状态
        z_proposed = self.propose_new_state(z_current)
        
        # 转换到概率空间
        fan_current = self.softmax_transform(z_current)
        fan_proposed = self.softmax_transform(z_proposed)
        
        # 计算当前状态的后验概率
        log_likelihood_current = likelihood_func(fan_current, judge_share, eliminated_idx)
        log_prior_current = self.calculate_log_prior(fan_current)
        log_posterior_current = log_likelihood_current + log_prior_current
        
        # 计算候选状态的后验概率
        log_likelihood_proposed = likelihood_func(fan_proposed, judge_share, eliminated_idx)
        log_prior_proposed = self.calculate_log_prior(fan_proposed)
        log_posterior_proposed = log_likelihood_proposed + log_prior_proposed
        
        # Metropolis-Hastings接受概率
        log_acceptance_ratio = log_posterior_proposed - log_posterior_current
        
        # 决定是否接受
        if log_acceptance_ratio > 0 or np.random.rand() < np.exp(log_acceptance_ratio):
            return z_proposed, True
        else:
            return z_current, False
    
    def sample_week(self, judge_share: np.ndarray,
                   eliminated_idx: int,
                   voting_method: str,
                   judge_ranks: np.ndarray = None) -> np.ndarray:
        """
        对单周进行MCMC采样估算粉丝投票
        
        Args:
            judge_share: 评委得分份额（已归一化）
            eliminated_idx: 被淘汰选手在存活者中的索引
            voting_method: 投票方法 ('rank', 'percentage', 'rank_bottom2')
            judge_ranks: 预计算的评委排名（可选，用于排名法）
            
        Returns:
            粉丝投票份额的后验样本 (n_samples × n_contestants)
        """
        n_contestants = len(judge_share)
        
        # 选择似然函数
        if voting_method == 'rank':
            likelihood_func = lambda f, j, e: self.calculate_log_likelihood_rank(f, j, e, False, judge_ranks)
        elif voting_method == 'percentage':
            likelihood_func = lambda f, j, e: self.calculate_log_likelihood_percentage(f, j, e)
        elif voting_method == 'rank_bottom2':
            likelihood_func = lambda f, j, e: self.calculate_log_likelihood_rank(f, j, e, True, judge_ranks)
        else:
            raise ValueError(f"未知的投票方法: {voting_method}")
        
        # 初始化无约束向量（从均匀分布开始）
        z_current = np.zeros(n_contestants)
        
        # 寻找满足约束的初始点
        found_valid_init = False
        for _ in range(5000):
            z_test = np.random.randn(n_contestants) * 0.5
            fan_test = self.softmax_transform(z_test)
            log_lik = likelihood_func(fan_test, judge_share, eliminated_idx)
            if log_lik > -np.inf:
                z_current = z_test
                found_valid_init = True
                break
        
        if not found_valid_init:
            warnings.warn("无法找到满足约束的初始点，采样可能不稳定")
        
        # MCMC采样
        samples = []
        accepted_burn_in = 0  # burn-in期间的接受次数
        accepted_sampling = 0  # 采样期间的接受次数
        
        for i in range(self.n_iterations):
            z_current, is_accepted = self.metropolis_hastings_step(
                z_current, judge_share, eliminated_idx, likelihood_func
            )
            
            if i < self.burn_in:
                if is_accepted:
                    accepted_burn_in += 1
            else:
                if is_accepted:
                    accepted_sampling += 1
                # Burn-in后开始保存样本（并进行抽稀）
                if (i - self.burn_in) % self.thinning == 0:
                    fan_votes = self.softmax_transform(z_current)
                    samples.append(fan_votes)
        
        # 只统计采样阶段的接受率
        n_sampling_iterations = self.n_iterations - self.burn_in
        self.acceptance_rate = accepted_sampling / n_sampling_iterations if n_sampling_iterations > 0 else 0.0
        self.samples = np.array(samples)
        
        return self.samples
    
    def sample_week_finale(self, judge_share: np.ndarray,
                          finale_rankings: list,
                          voting_method: str) -> np.ndarray:
        """
        对决赛周进行MCMC采样，约束最终排名
        
        Args:
            judge_share: 评委得分份额（已归一化）
            finale_rankings: 决赛排名列表 [{'survivor_idx': idx, 'place': 排名}, ...]
            voting_method: 投票方法
            
        Returns:
            粉丝投票份额的后验样本 (n_samples × n_contestants)
        """
        n_contestants = len(judge_share)
        
        # 解析排名约束：从好到差排列的选手索引
        sorted_rankings = sorted(finale_rankings, key=lambda x: x['final_place'])
        ranking_order = [r['survivor_idx'] for r in sorted_rankings]  # 按名次从1到n排序
        
        # 定义似然函数：粉丝投票需要产生正确的最终排名
        def calculate_log_likelihood_finale(fan_votes: np.ndarray) -> float:
            """计算满足决赛排名约束的似然度"""
            if voting_method in ['rank', 'rank_bottom2']:
                n = len(fan_votes)
                judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
                fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
                combined = judge_rank + fan_rank  # 越小越好
                predicted_order = np.argsort(combined)
            else:  # percentage
                combined = 0.5 * judge_share + 0.5 * fan_votes
                predicted_order = np.argsort(-combined)  # 从高到低
            
            # 检查预测排名是否与实际排名一致
            predicted_order_list = list(predicted_order)
            
            # 计算排名差异的惩罚
            penalty = 0.0
            for i, correct_idx in enumerate(ranking_order):
                if i < len(predicted_order_list):
                    predicted_position = predicted_order_list.index(correct_idx) if correct_idx in predicted_order_list else i
                    penalty += (predicted_position - i) ** 2
            
            # 完全匹配时penalty=0，给予高似然度；否则惩罚
            if penalty == 0:
                return 0.0  # log(1) = 0
            else:
                return -penalty * 5.0  # 强惩罚偏离正确排名的情况
        
        # 初始化无约束向量
        z_current = np.zeros(n_contestants)
        
        # 找一个满足约束的初始点
        for _ in range(1000):
            z_test = np.random.randn(n_contestants) * 0.5
            fan_test = self.softmax_transform(z_test)
            if calculate_log_likelihood_finale(fan_test) == 0.0:
                z_current = z_test
                break
        
        # MCMC采样
        samples = []
        accepted_burn_in = 0
        accepted_sampling = 0
        
        for i in range(self.n_iterations):
            # 提议新状态
            z_proposed = z_current + np.random.randn(n_contestants) * self.proposal_sigma
            
            fan_current = self.softmax_transform(z_current)
            fan_proposed = self.softmax_transform(z_proposed)
            
            log_lik_current = calculate_log_likelihood_finale(fan_current)
            log_lik_proposed = calculate_log_likelihood_finale(fan_proposed)
            
            # 先验（均匀）
            log_prior_current = -0.5 * np.sum(z_current ** 2)
            log_prior_proposed = -0.5 * np.sum(z_proposed ** 2)
            
            log_posterior_current = log_lik_current + log_prior_current
            log_posterior_proposed = log_lik_proposed + log_prior_proposed
            
            log_acceptance_ratio = log_posterior_proposed - log_posterior_current
            
            is_accepted = log_acceptance_ratio > 0 or np.random.rand() < np.exp(log_acceptance_ratio)
            if is_accepted:
                z_current = z_proposed
            
            if i < self.burn_in:
                if is_accepted:
                    accepted_burn_in += 1
            else:
                if is_accepted:
                    accepted_sampling += 1
                if (i - self.burn_in) % self.thinning == 0:
                    fan_votes = self.softmax_transform(z_current)
                    samples.append(fan_votes)
        
        n_sampling_iterations = self.n_iterations - self.burn_in
        self.acceptance_rate = accepted_sampling / n_sampling_iterations if n_sampling_iterations > 0 else 0.0
        self.samples = np.array(samples)
        
        return self.samples
    
    def calculate_hpdi(self, samples: np.ndarray, credibility: float = 0.95) -> np.ndarray:
        """
        计算最高后验密度区间 (HPDI)
        
        Args:
            samples: 后验样本 (n_samples × n_contestants)
            credibility: 可信度（默认95%）
            
        Returns:
            HPDI区间 (n_contestants × 2)，每行为[lower, upper]
        """
        n_contestants = samples.shape[1]
        hpdi = np.zeros((n_contestants, 2))
        
        for i in range(n_contestants):
            contestant_samples = np.sort(samples[:, i])
            n = len(contestant_samples)
            
            # 计算区间长度
            interval_size = int(np.ceil(credibility * n))
            if interval_size >= n:
                interval_size = n - 1
            
            n_intervals = n - interval_size
            if n_intervals <= 0:
                hpdi[i, 0] = contestant_samples[0]
                hpdi[i, 1] = contestant_samples[-1]
                continue
            
            interval_width = contestant_samples[interval_size:] - contestant_samples[:n_intervals]
            
            # 找到最窄的区间
            min_idx = np.argmin(interval_width)
            hpdi[i, 0] = contestant_samples[min_idx]
            hpdi[i, 1] = contestant_samples[min_idx + interval_size]
        
        return hpdi


if __name__ == "__main__":
    # 测试代码
    print("MCMC采样器测试\n")
    
    # 模拟一个简单场景：4名选手，第3名被淘汰
    judge_share = np.array([0.28, 0.32, 0.20, 0.20])
    eliminated_idx = 2  # 第3名选手（0-indexed）
    
    sampler = MCMCSampler(n_iterations=5000, burn_in=1000, thinning=5)
    
    print("测试百分比法:")
    samples = sampler.sample_week(judge_share, eliminated_idx, 'percentage')
    print(f"采样完成: {samples.shape[0]} 个样本")
    print(f"接受率: {sampler.acceptance_rate:.2%}")
    print(f"粉丝投票估计均值: {samples.mean(axis=0)}")
    print(f"粉丝投票标准差: {samples.std(axis=0)}")
    
    hpdi = sampler.calculate_hpdi(samples)
    print(f"\n95% HPDI区间:")
    for i in range(len(judge_share)):
        print(f"  选手{i+1}: [{hpdi[i,0]:.3f}, {hpdi[i,1]:.3f}]")
