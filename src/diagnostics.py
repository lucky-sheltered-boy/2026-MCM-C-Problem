"""
模型诊断与验证模块 (Model Diagnostics and Validation)
包含收敛性诊断、确定性度量和一致性验证
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


class ModelDiagnostics:
    """模型诊断工具类"""
    
    @staticmethod
    def gelman_rubin_statistic(chains: List[np.ndarray]) -> np.ndarray:
        """
        计算Gelman-Rubin统计量 (R-hat) 评估多链收敛性
        
        Args:
            chains: 多条链的样本列表，每条链形状为 (n_samples, n_params)
            
        Returns:
            每个参数的R-hat值（应小于1.1表示收敛）
        """
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]
        
        # 确保所有链长度一致
        assert all(chain.shape == (n_samples, n_params) for chain in chains), \
            "所有链必须有相同的形状"
        
        # 计算链内方差
        within_chain_var = np.mean([np.var(chain, axis=0, ddof=1) for chain in chains], axis=0)
        
        # 计算链间方差
        chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
        overall_mean = np.mean(chain_means, axis=0)
        between_chain_var = n_samples * np.var(chain_means, axis=0, ddof=1)
        
        # 计算边际后验方差的估计
        marginal_post_var = ((n_samples - 1) / n_samples) * within_chain_var + \
                           (1 / n_samples) * between_chain_var
        
        # Gelman-Rubin统计量
        r_hat = np.sqrt(marginal_post_var / (within_chain_var + 1e-10))
        
        return r_hat
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """
        计算有效样本量 (ESS)
        
        Args:
            samples: 样本数组 (n_samples, n_params)
            max_lag: 最大滞后阶数
            
        Returns:
            每个参数的有效样本量
        """
        n_samples, n_params = samples.shape
        ess = np.zeros(n_params)
        
        for i in range(n_params):
            # 计算自相关函数
            param_samples = samples[:, i]
            param_centered = param_samples - np.mean(param_samples)
            
            autocorr = np.correlate(param_centered, param_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # 计算有效样本量
            # ESS = n / (1 + 2 * sum(rho_k))
            max_lag = min(max_lag, len(autocorr) - 1)
            autocorr_sum = np.sum(autocorr[1:max_lag+1])
            ess[i] = n_samples / (1 + 2 * autocorr_sum)
        
        return ess
    
    @staticmethod
    def posterior_predictive_check(samples: np.ndarray,
                                   judge_share: np.ndarray,
                                   eliminated_idx: int,
                                   voting_method: str) -> float:
        """
        后验预测检查：使用样本重新预测淘汰结果
        
        Args:
            samples: 粉丝投票的后验样本 (n_samples, n_contestants)
            judge_share: 评委得分份额
            eliminated_idx: 实际被淘汰选手的索引
            voting_method: 投票方法
            
        Returns:
            预测正确的比例（一致性分数）
        """
        n_samples = samples.shape[0]
        correct_predictions = 0
        
        for fan_votes in samples:
            predicted_eliminated = ModelDiagnostics._predict_elimination(
                fan_votes, judge_share, voting_method
            )
            
            if voting_method == 'rank_bottom2':
                # Bottom Two机制：只要在底部两名中即可
                if eliminated_idx in predicted_eliminated:
                    correct_predictions += 1
            else:
                # 其他方法：必须精确预测
                if predicted_eliminated == eliminated_idx:
                    correct_predictions += 1
        
        return correct_predictions / n_samples
    
    @staticmethod
    def _predict_elimination(fan_votes: np.ndarray,
                           judge_share: np.ndarray,
                           voting_method: str) -> int or List[int]:
        """
        根据投票方法预测被淘汰的选手
        
        Args:
            fan_votes: 粉丝投票份额
            judge_share: 评委得分份额
            voting_method: 投票方法
            
        Returns:
            被淘汰选手的索引（Bottom Two返回列表）
        """
        if voting_method == 'rank' or voting_method == 'rank_bottom2':
            # 排名法
            n = len(fan_votes)
            judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
            fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
            total_rank = judge_rank + fan_rank
            
            if voting_method == 'rank_bottom2':
                # 返回排名最差的两位
                return list(np.argsort(total_rank)[-2:])
            else:
                # 返回排名最差的一位
                return int(np.argmax(total_rank))
        
        elif voting_method == 'percentage':
            # 百分比法
            combined_score = 0.5 * judge_share + 0.5 * fan_votes
            return int(np.argmin(combined_score))
        
        else:
            raise ValueError(f"未知的投票方法: {voting_method}")


class CertaintyMetrics:
    """确定性度量类"""
    
    @staticmethod
    def calculate_hpdi_width(samples: np.ndarray, credibility: float = 0.95) -> np.ndarray:
        """
        计算HPDI区间宽度（作为不确定性的度量）
        
        Args:
            samples: 后验样本 (n_samples, n_contestants)
            credibility: 可信度
            
        Returns:
            每位选手的HPDI宽度
        """
        n_contestants = samples.shape[1]
        widths = np.zeros(n_contestants)
        
        for i in range(n_contestants):
            contestant_samples = np.sort(samples[:, i])
            n = len(contestant_samples)
            
            interval_size = int(np.ceil(credibility * n))
            if interval_size >= n:
                interval_size = n - 1
            
            n_intervals = n - interval_size
            if n_intervals <= 0:
                widths[i] = contestant_samples[-1] - contestant_samples[0]
                continue
                
            interval_width = contestant_samples[interval_size:] - contestant_samples[:n_intervals]
            
            min_idx = np.argmin(interval_width)
            widths[i] = interval_width[min_idx]
        
        return widths
    
    @staticmethod
    def calculate_posterior_std(samples: np.ndarray) -> np.ndarray:
        """
        计算后验标准差
        
        Args:
            samples: 后验样本 (n_samples, n_contestants)
            
        Returns:
            每位选手的后验标准差
        """
        return np.std(samples, axis=0)
    
    @staticmethod
    def calculate_coefficient_of_variation(samples: np.ndarray) -> np.ndarray:
        """
        计算变异系数 (CV = std / mean)
        
        Args:
            samples: 后验样本 (n_samples, n_contestants)
            
        Returns:
            每位选手的变异系数
        """
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        return stds / (means + 1e-10)
    
    @staticmethod
    def identify_outliers(samples: np.ndarray, 
                         prior_mean: np.ndarray,
                         prior_std: np.ndarray,
                         threshold: float = 3.0) -> Dict:
        """
        识别统计异常值（如Bobby Bones现象）
        
        Args:
            samples: 后验样本
            prior_mean: 先验均值
            prior_std: 先验标准差
            threshold: z-score阈值
            
        Returns:
            异常信息字典
        """
        posterior_mean = np.mean(samples, axis=0)
        
        # 计算z-score
        z_scores = (posterior_mean - prior_mean) / (prior_std + 1e-10)
        
        outliers = {}
        for i, z in enumerate(z_scores):
            if np.abs(z) > threshold:
                outliers[i] = {
                    'z_score': z,
                    'posterior_mean': posterior_mean[i],
                    'prior_mean': prior_mean[i],
                    'deviation': 'extreme_high' if z > 0 else 'extreme_low'
                }
        
        return outliers


class ConsistencyValidator:
    """一致性验证器"""
    
    def __init__(self):
        self.results = {}
    
    def validate_season(self, season_samples: Dict[int, np.ndarray],
                       season_data: Dict,
                       voting_method: str) -> Dict:
        """
        验证整个赛季的一致性
        
        Args:
            season_samples: {week: samples} 每周的后验样本
            season_data: 赛季数据（包含实际淘汰信息）
            voting_method: 投票方法
            
        Returns:
            一致性报告
        """
        consistency_scores = {}
        
        for week_num, week_data in season_data['weeks'].items():
            if week_num not in season_samples:
                continue
            
            samples = season_samples[week_num]
            judge_share = week_data['judge_share']
            eliminated_idx = week_data['eliminated_idx_in_survivors']
            
            # 后验预测检查
            ppc_score = ModelDiagnostics.posterior_predictive_check(
                samples, judge_share, eliminated_idx, voting_method
            )
            
            consistency_scores[week_num] = ppc_score
        
        # 计算平均一致性
        avg_consistency = np.mean(list(consistency_scores.values()))
        
        return {
            'weekly_scores': consistency_scores,
            'average_consistency': avg_consistency,
            'perfect_weeks': sum(1 for s in consistency_scores.values() if s > 0.95),
            'total_weeks': len(consistency_scores)
        }
    
    def compare_voting_methods(self, season_samples: Dict[int, np.ndarray],
                              season_data: Dict) -> Dict:
        """
        比较不同投票方法在同一数据上的表现
        
        Args:
            season_samples: 每周的后验样本
            season_data: 赛季数据
            
        Returns:
            不同方法的比较结果
        """
        methods = ['rank', 'percentage', 'rank_bottom2']
        results = {}
        
        for method in methods:
            consistency = self.validate_season(season_samples, season_data, method)
            results[method] = consistency
        
        return results


if __name__ == "__main__":
    # 测试代码
    print("模型诊断工具测试\n")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_contestants = 4
    
    # 模拟3条MCMC链
    chains = [
        np.random.dirichlet([2, 3, 1, 2], n_samples) for _ in range(3)
    ]
    
    # Gelman-Rubin统计量
    r_hat = ModelDiagnostics.gelman_rubin_statistic(chains)
    print(f"Gelman-Rubin R-hat: {r_hat}")
    print(f"收敛状态: {'✓ 已收敛' if np.all(r_hat < 1.1) else '✗ 未收敛'}")
    
    # 有效样本量
    ess = ModelDiagnostics.effective_sample_size(chains[0])
    print(f"\n有效样本量: {ess}")
    
    # 确定性度量
    hpdi_widths = CertaintyMetrics.calculate_hpdi_width(chains[0])
    print(f"\nHPDI宽度: {hpdi_widths}")
    
    cv = CertaintyMetrics.calculate_coefficient_of_variation(chains[0])
    print(f"变异系数: {cv}")
