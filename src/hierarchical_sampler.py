"""
层次贝叶斯MCMC采样器 (Hierarchical Bayesian MCMC Sampler)
实现基本盘+表现票的分解模型

模型结构:
    f_{i,w} = λ·α_i + (1-λ)·softmax(β_{i,w} + γ·judge_score_{i,w})
    
其中:
    α_i: 选手i的基本盘份额 (整个赛季固定)
    β_{i,w}: 选手i在第w周的表现波动
    λ: 基本盘权重 (默认0.7)
    γ: 评委分对表现票的影响系数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax
from dataclasses import dataclass


@dataclass
class SeasonData:
    """赛季数据结构"""
    season: int
    n_contestants: int
    contestant_names: List[str]
    weeks: List[Dict]  # 每周的数据
    voting_method: str
    
    # 选手存活矩阵: active[w][i] = True 表示选手i在第w周存活
    active_matrix: np.ndarray  # (n_weeks, n_contestants)
    
    # 评委分矩阵
    judge_scores: np.ndarray  # (n_weeks, n_contestants)
    
    # 淘汰信息
    eliminations: List[Dict]  # 每周的淘汰信息


class HierarchicalMCMCSampler:
    """层次贝叶斯MCMC采样器"""
    
    def __init__(self, 
                 n_iterations: int = 15000,
                 burn_in: int = 5000,
                 thinning: int = 5,
                 base_weight: float = 0.7,
                 proposal_sigma_alpha: float = 0.1,
                 proposal_sigma_beta: float = 0.2,
                 smoothness_prior_std: float = 0.1):
        """
        初始化层次采样器
        
        Args:
            n_iterations: 总迭代次数
            burn_in: Burn-in期
            thinning: 抽稀间隔
            base_weight: 基本盘权重λ (0.5-0.9)
            proposal_sigma_alpha: α提议分布标准差
            proposal_sigma_beta: β提议分布标准差
            smoothness_prior_std: 平滑性先验的标准差
        """
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thinning = thinning
        self.base_weight = base_weight
        self.proposal_sigma_alpha = proposal_sigma_alpha
        self.proposal_sigma_beta = proposal_sigma_beta
        self.smoothness_prior_std = smoothness_prior_std
        
        # 采样结果
        self.alpha_samples = None  # 基本盘样本
        self.beta_samples = None   # 表现票样本
        self.fan_votes_samples = None  # 合成的粉丝投票
        self.acceptance_rate = 0.0
        
    def _softmax_transform(self, z: np.ndarray) -> np.ndarray:
        """Softmax变换"""
        return softmax(z)
    
    def _compute_fan_votes(self, alpha: np.ndarray, beta: np.ndarray, 
                           active_mask: np.ndarray) -> np.ndarray:
        """
        计算合成的粉丝投票份额
        
        Args:
            alpha: 基本盘份额 (n_contestants,)
            beta: 表现票无约束向量 (n_contestants,)
            active_mask: 存活掩码 (n_contestants,) boolean
            
        Returns:
            粉丝投票份额 (仅存活选手，已归一化)
        """
        n_active = np.sum(active_mask)
        
        # 提取存活选手的基本盘
        alpha_active = alpha[active_mask]
        alpha_active = alpha_active / (np.sum(alpha_active) + 1e-10)  # 重新归一化
        
        # 表现票通过softmax归一化
        beta_active = beta[active_mask]
        performance_votes = self._softmax_transform(beta_active)
        
        # 合成
        fan_votes = self.base_weight * alpha_active + (1 - self.base_weight) * performance_votes
        
        return fan_votes
    
    def _log_likelihood_week(self, fan_votes: np.ndarray, 
                             judge_share: np.ndarray,
                             eliminated_idx: int,
                             voting_method: str) -> float:
        """计算单周的对数似然"""
        n = len(fan_votes)
        
        if voting_method == 'percentage':
            combined = 0.5 * judge_share + 0.5 * fan_votes
            scores_sorted = np.sort(combined)
            actual_score = combined[eliminated_idx]
            # 被淘汰者应该是得分最低的
            # 使用更强的似然约束
            min_score = scores_sorted[0]
            if actual_score == min_score:
                return 0.0  # 完美预测
            else:
                # 惩罚力度与排名差距成正比
                rank_actual = np.sum(combined < actual_score)  # 有多少人比他差
                return -10.0 * rank_actual  # 强惩罚
        else:  # rank or rank_bottom2
            judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
            fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
            total_rank = judge_rank + fan_rank
            
            # 被淘汰者应该总排名最差（数值最大）
            actual_rank = total_rank[eliminated_idx]
            max_rank = np.max(total_rank)
            
            if actual_rank == max_rank:
                return 0.0  # 完美预测
            else:
                # 惩罚
                rank_diff = max_rank - actual_rank
                return -10.0 * rank_diff
    
    def _log_prior_alpha(self, z_alpha: np.ndarray) -> float:
        """基本盘的先验（近似均匀Dirichlet）"""
        return -0.5 * np.sum(z_alpha ** 2) * 0.1  # 弱先验
    
    def _log_prior_beta(self, beta_all_weeks: np.ndarray, 
                        active_matrix: np.ndarray) -> float:
        """
        表现票的先验：鼓励跨周平滑
        
        Args:
            beta_all_weeks: (n_weeks, n_contestants)
            active_matrix: (n_weeks, n_contestants)
        """
        log_prior = 0.0
        n_weeks, n_contestants = beta_all_weeks.shape
        
        # 每个选手的β应该随时间平滑变化
        for i in range(n_contestants):
            active_weeks = np.where(active_matrix[:, i])[0]
            if len(active_weeks) < 2:
                continue
            
            for j in range(len(active_weeks) - 1):
                w1, w2 = active_weeks[j], active_weeks[j + 1]
                diff = beta_all_weeks[w2, i] - beta_all_weeks[w1, i]
                log_prior -= 0.5 * (diff / self.smoothness_prior_std) ** 2
        
        # β本身的先验（均值为0）
        log_prior -= 0.5 * np.sum(beta_all_weeks ** 2) * 0.05
        
        return log_prior
    
    def sample_season(self, season_data: SeasonData) -> Dict:
        """
        对整个赛季进行层次MCMC采样
        
        Args:
            season_data: 赛季数据
            
        Returns:
            采样结果字典
        """
        n_contestants = season_data.n_contestants
        n_weeks = len(season_data.weeks)
        active_matrix = season_data.active_matrix
        
        # 初始化参数
        z_alpha = np.zeros(n_contestants)  # 基本盘的无约束参数
        beta = np.zeros((n_weeks, n_contestants))  # 表现票参数
        
        # 存储样本
        alpha_samples = []
        beta_samples = []
        fan_votes_by_week = {w: [] for w in range(n_weeks)}
        
        accepted = 0
        
        for iteration in range(self.n_iterations):
            # === 更新 α (基本盘) ===
            z_alpha_proposed = z_alpha + np.random.randn(n_contestants) * self.proposal_sigma_alpha
            
            # 计算当前和提议的对数后验
            log_post_current = self._compute_log_posterior(
                z_alpha, beta, season_data
            )
            log_post_proposed = self._compute_log_posterior(
                z_alpha_proposed, beta, season_data
            )
            
            # MH接受步
            if np.log(np.random.rand()) < (log_post_proposed - log_post_current):
                z_alpha = z_alpha_proposed
                accepted += 1
            
            # === 更新 β (表现票) ===
            for w in range(n_weeks):
                week_info = season_data.weeks[w]
                if week_info.get('is_no_elimination', False) and not week_info.get('is_finale', False):
                    continue
                
                active_mask = active_matrix[w]
                beta_w_proposed = beta[w].copy()
                beta_w_proposed[active_mask] += np.random.randn(np.sum(active_mask)) * self.proposal_sigma_beta
                
                beta_proposed = beta.copy()
                beta_proposed[w] = beta_w_proposed
                
                log_post_current_w = self._compute_log_posterior(z_alpha, beta, season_data)
                log_post_proposed_w = self._compute_log_posterior(z_alpha, beta_proposed, season_data)
                
                if np.log(np.random.rand()) < (log_post_proposed_w - log_post_current_w):
                    beta = beta_proposed
            
            # 保存样本
            if iteration >= self.burn_in and (iteration - self.burn_in) % self.thinning == 0:
                alpha = self._softmax_transform(z_alpha)
                alpha_samples.append(alpha.copy())
                beta_samples.append(beta.copy())
                
                # 计算每周的粉丝投票
                for w in range(n_weeks):
                    active_mask = active_matrix[w]
                    if np.sum(active_mask) > 0:
                        fv = self._compute_fan_votes(alpha, beta[w], active_mask)
                        fan_votes_by_week[w].append(fv)
        
        self.acceptance_rate = accepted / self.n_iterations
        self.alpha_samples = np.array(alpha_samples)
        self.beta_samples = np.array(beta_samples)
        
        # 整理结果
        results = {
            'alpha_samples': self.alpha_samples,
            'beta_samples': self.beta_samples,
            'alpha_mean': np.mean(self.alpha_samples, axis=0),
            'alpha_std': np.std(self.alpha_samples, axis=0),
            'fan_votes_by_week': {},
            'acceptance_rate': self.acceptance_rate,
            'contestant_names': season_data.contestant_names
        }
        
        for w in range(n_weeks):
            if fan_votes_by_week[w]:
                samples = np.array(fan_votes_by_week[w])
                results['fan_votes_by_week'][w] = {
                    'samples': samples,
                    'mean': np.mean(samples, axis=0),
                    'std': np.std(samples, axis=0)
                }
        
        return results
    
    def _compute_log_posterior(self, z_alpha: np.ndarray, 
                               beta: np.ndarray,
                               season_data: SeasonData) -> float:
        """计算完整的对数后验"""
        alpha = self._softmax_transform(z_alpha)
        
        # 先验
        log_prior = self._log_prior_alpha(z_alpha)
        log_prior += self._log_prior_beta(beta, season_data.active_matrix)
        
        # 似然
        log_likelihood = 0.0
        
        for w, week_info in enumerate(season_data.weeks):
            if week_info.get('is_no_elimination', False) and not week_info.get('is_finale', False):
                continue
            
            active_mask = season_data.active_matrix[w]
            if np.sum(active_mask) == 0:
                continue
            
            fan_votes = self._compute_fan_votes(alpha, beta[w], active_mask)
            judge_share = week_info['judge_share']
            
            # 处理淘汰
            eliminated_indices = week_info.get('eliminated_indices_in_survivors', [])
            
            for elim_idx in eliminated_indices:
                if elim_idx is not None and 0 <= elim_idx < len(fan_votes):
                    log_likelihood += self._log_likelihood_week(
                        fan_votes, judge_share, elim_idx, season_data.voting_method
                    )
        
        return log_prior + log_likelihood


def prepare_season_data(processed_data: Dict, season: int) -> SeasonData:
    """
    从处理后的数据准备SeasonData结构
    
    Args:
        processed_data: data_loader处理后的数据
        season: 赛季编号
        
    Returns:
        SeasonData实例
    """
    season_info = processed_data[season]
    weeks_data = season_info['weeks']
    
    # 获取所有选手名单（从第一周）
    first_week = min(weeks_data.keys())
    all_contestants = list(weeks_data[first_week]['survivor_names'])
    n_contestants = len(all_contestants)
    n_weeks = len(weeks_data)
    
    # 构建存活矩阵和评委分矩阵
    active_matrix = np.zeros((n_weeks, n_contestants), dtype=bool)
    judge_scores = np.zeros((n_weeks, n_contestants))
    
    weeks_list = []
    
    for w_idx, week_num in enumerate(sorted(weeks_data.keys())):
        week = weeks_data[week_num]
        survivors = week['survivor_names']
        
        for i, name in enumerate(all_contestants):
            if name in survivors:
                active_matrix[w_idx, i] = True
                surv_idx = survivors.index(name)
                if surv_idx < len(week['judge_share']):
                    judge_scores[w_idx, i] = week['judge_share'][surv_idx]
        
        weeks_list.append(week)
    
    return SeasonData(
        season=season,
        n_contestants=n_contestants,
        contestant_names=all_contestants,
        weeks=weeks_list,
        voting_method=season_info['voting_method'],
        active_matrix=active_matrix,
        judge_scores=judge_scores,
        eliminations=[]
    )


if __name__ == "__main__":
    print("层次贝叶斯MCMC采样器测试")
    print("=" * 50)
    
    # 模拟简单场景
    sampler = HierarchicalMCMCSampler(
        n_iterations=5000,
        burn_in=1000,
        base_weight=0.7
    )
    
    print(f"基本盘权重 λ = {sampler.base_weight}")
    print(f"迭代次数 = {sampler.n_iterations}")
    print("\n使用 main_hierarchical.py 进行完整测试")
