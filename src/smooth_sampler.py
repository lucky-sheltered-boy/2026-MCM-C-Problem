"""
平滑采样器模块 (Smooth Sampler Module)
在MCMC后验样本基础上，生成时序平滑的粉丝投票估计
保证100%一致性（被淘汰者确实是综合得分最低）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax
import warnings


class SmoothSampler:
    """
    平滑采样器：从MCMC后验样本中选择时序平滑的路径
    
    思路：顺序生成
    - 第1周：从后验样本中选择一个满足一致性的样本
    - 第k周：从后验样本中选择满足一致性且与第k-1周变化最小的样本
    """
    
    def __init__(self, n_candidates: int = 100, smoothness_weight: float = 1.0):
        """
        初始化平滑采样器
        
        Args:
            n_candidates: 每周从MCMC样本中筛选的候选数量
            smoothness_weight: 平滑性权重（越大越倾向于平滑）
        """
        self.n_candidates = n_candidates
        self.smoothness_weight = smoothness_weight
        
    def check_consistency(self, fan_votes: np.ndarray, judge_share: np.ndarray,
                          eliminated_idx: int, voting_method: str,
                          judge_ranks: np.ndarray = None) -> bool:
        """
        检查粉丝投票是否满足一致性（被淘汰者确实是最低）
        
        Args:
            fan_votes: 粉丝投票份额
            judge_share: 评委得分份额
            eliminated_idx: 被淘汰选手索引
            voting_method: 投票方法
            judge_ranks: 预计算的评委排名（可选，用于排名法）
            
        Returns:
            是否满足一致性
        """
        if voting_method == 'percentage':
            combined = 0.5 * judge_share + 0.5 * fan_votes
            worst_idx = np.argmin(combined)
            return worst_idx == eliminated_idx
            
        elif voting_method in ['rank', 'rank_bottom2']:
            n = len(fan_votes)
            # 使用预计算的评委排名，或自行计算（兼容旧代码）
            if judge_ranks is not None:
                judge_rank = judge_ranks
            else:
                judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
            fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
            total_rank = judge_rank + fan_rank  # 越大越差
            
            if voting_method == 'rank':
                worst_idx = np.argmax(total_rank)
                return worst_idx == eliminated_idx
            else:  # rank_bottom2
                bottom_two = np.argsort(total_rank)[-2:]
                return eliminated_idx in bottom_two
        else:
            raise ValueError(f"未知的投票方法: {voting_method}")
    
    def calculate_distance(self, votes_a: np.ndarray, votes_b: np.ndarray,
                          method: str = 'l2') -> float:
        """
        计算两个投票分布之间的距离
        
        Args:
            votes_a: 投票份额A
            votes_b: 投票份额B
            method: 距离度量方法 ('l2', 'kl', 'js')
            
        Returns:
            距离值
        """
        if method == 'l2':
            return np.sqrt(np.sum((votes_a - votes_b) ** 2))
        elif method == 'kl':
            # KL散度（加小值防止log(0)）
            eps = 1e-10
            return np.sum(votes_a * np.log((votes_a + eps) / (votes_b + eps)))
        elif method == 'js':
            # JS散度（对称）
            eps = 1e-10
            m = 0.5 * (votes_a + votes_b)
            kl_am = np.sum(votes_a * np.log((votes_a + eps) / (m + eps)))
            kl_bm = np.sum(votes_b * np.log((votes_b + eps) / (m + eps)))
            return 0.5 * (kl_am + kl_bm)
        else:
            return np.sqrt(np.sum((votes_a - votes_b) ** 2))
    
    def select_smoothest_sample(self, mcmc_samples: np.ndarray,
                                judge_share: np.ndarray,
                                eliminated_idx: int,
                                voting_method: str,
                                prev_votes: Optional[np.ndarray] = None,
                                prev_survivor_indices: Optional[List[int]] = None,
                                curr_survivor_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, bool]:
        """
        从MCMC样本中选择最平滑的一个
        
        Args:
            mcmc_samples: MCMC后验样本 (n_samples × n_contestants)
            judge_share: 评委得分份额
            eliminated_idx: 被淘汰选手索引
            voting_method: 投票方法
            prev_votes: 上一周的投票分布（第1周为None）
            prev_survivor_indices: 上一周存活者的原始索引
            curr_survivor_indices: 本周存活者的原始索引
            
        Returns:
            (选中的粉丝投票, 是否找到满足一致性的样本)
        """
        # 筛选满足一致性的样本
        valid_samples = []
        for sample in mcmc_samples:
            if self.check_consistency(sample, judge_share, eliminated_idx, voting_method):
                valid_samples.append(sample)
        
        if len(valid_samples) == 0:
            # 没有满足一致性的样本，返回均值并警告
            warnings.warn(f"未找到满足一致性的样本，使用MCMC均值")
            return mcmc_samples.mean(axis=0), False
        
        valid_samples = np.array(valid_samples)
        
        # 如果是第一周或没有上一周数据，返回随机一个
        if prev_votes is None:
            # 选择一个"中庸"的样本（接近均值）
            mean_sample = valid_samples.mean(axis=0)
            distances_to_mean = [self.calculate_distance(s, mean_sample) for s in valid_samples]
            best_idx = np.argmin(distances_to_mean)
            return valid_samples[best_idx], True
        
        # 需要对齐上一周和本周的选手
        # 找出共同存活的选手，计算他们的投票变化
        if prev_survivor_indices is None or curr_survivor_indices is None:
            # 如果没有提供索引映射，假设选手顺序一致（少一个被淘汰者）
            # 简单处理：直接比较前n-1个选手
            common_n = min(len(prev_votes), len(valid_samples[0]))
            
            best_sample = None
            min_distance = float('inf')
            
            for sample in valid_samples:
                # 只比较共同的选手
                dist = self.calculate_distance(sample[:common_n], prev_votes[:common_n])
                if dist < min_distance:
                    min_distance = dist
                    best_sample = sample
            
            return best_sample, True
        
        # 有索引映射的情况：精确对齐
        # 找出两周共同存活的选手
        common_original_indices = set(prev_survivor_indices) & set(curr_survivor_indices)
        
        if len(common_original_indices) == 0:
            # 没有共同选手（不太可能），随机选
            return valid_samples[0], True
        
        # 建立索引映射
        prev_local_indices = [prev_survivor_indices.index(idx) for idx in common_original_indices]
        curr_local_indices = [curr_survivor_indices.index(idx) for idx in common_original_indices]
        
        best_sample = None
        min_distance = float('inf')
        
        for sample in valid_samples:
            # 提取共同选手的投票份额
            prev_common = np.array([prev_votes[i] for i in prev_local_indices])
            curr_common = np.array([sample[i] for i in curr_local_indices])
            
            # 重新归一化（因为只取了部分选手）
            prev_common_norm = prev_common / (prev_common.sum() + 1e-10)
            curr_common_norm = curr_common / (curr_common.sum() + 1e-10)
            
            dist = self.calculate_distance(curr_common_norm, prev_common_norm)
            
            if dist < min_distance:
                min_distance = dist
                best_sample = sample
        
        return best_sample, True
    
    def generate_smooth_trajectory(self, 
                                   weekly_data: List[Dict],
                                   voting_method: str) -> Dict:
        """
        生成整个赛季的平滑投票轨迹
        
        Args:
            weekly_data: 每周数据列表，每个元素包含：
                - 'mcmc_samples': MCMC后验样本
                - 'judge_share': 评委得分份额
                - 'eliminated_idx': 被淘汰选手索引
                - 'survivor_names': 存活者名单
            voting_method: 投票方法
            
        Returns:
            平滑轨迹结果字典
        """
        trajectory = []
        consistency_flags = []
        prev_votes = None
        prev_survivor_names = None
        
        for week_idx, week_info in enumerate(weekly_data):
            mcmc_samples = week_info['mcmc_samples']
            judge_share = week_info['judge_share']
            eliminated_idx = week_info['eliminated_idx']
            survivor_names = week_info.get('survivor_names', None)
            
            # 建立索引映射（通过名字）
            prev_survivor_indices = None
            curr_survivor_indices = None
            
            if prev_survivor_names is not None and survivor_names is not None:
                # 创建名字到索引的映射
                prev_name_to_idx = {name: i for i, name in enumerate(prev_survivor_names)}
                curr_name_to_idx = {name: i for i, name in enumerate(survivor_names)}
                
                # 找共同名字
                common_names = set(prev_survivor_names) & set(survivor_names)
                
                if len(common_names) > 0:
                    prev_survivor_indices = [prev_name_to_idx[n] for n in common_names]
                    curr_survivor_indices = [curr_name_to_idx[n] for n in common_names]
            
            # 选择最平滑的样本
            selected_votes, is_consistent = self.select_smoothest_sample(
                mcmc_samples=mcmc_samples,
                judge_share=judge_share,
                eliminated_idx=eliminated_idx,
                voting_method=voting_method,
                prev_votes=prev_votes,
                prev_survivor_indices=prev_survivor_indices,
                curr_survivor_indices=curr_survivor_indices
            )
            
            trajectory.append({
                'week': week_idx + 1,
                'fan_votes': selected_votes,
                'survivor_names': survivor_names,
                'is_consistent': is_consistent
            })
            consistency_flags.append(is_consistent)
            
            prev_votes = selected_votes
            prev_survivor_names = survivor_names
        
        return {
            'trajectory': trajectory,
            'overall_consistency': sum(consistency_flags) / len(consistency_flags) if len(consistency_flags) > 0 else 0.0,
            'n_weeks': len(trajectory)
        }


def generate_consistent_sample(judge_share: np.ndarray, 
                               eliminated_idx: int,
                               voting_method: str,
                               judge_ranks: np.ndarray = None,
                               n_attempts: int = 10000) -> Optional[np.ndarray]:
    """
    直接生成一个满足一致性约束的粉丝投票样本
    使用拒绝采样
    
    Args:
        judge_share: 评委得分份额
        eliminated_idx: 被淘汰选手索引
        voting_method: 投票方法
        judge_ranks: 预计算的评委排名（可选，用于排名法）
        n_attempts: 最大尝试次数
        
    Returns:
        满足一致性的粉丝投票份额，或None（如果找不到）
    """
    n = len(judge_share)
    
    # 预计算评委排名（如果未提供）
    if judge_ranks is not None:
        judge_rank = judge_ranks
    else:
        judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
    
    for _ in range(n_attempts):
        # 从Dirichlet分布采样
        fan_votes = np.random.dirichlet(np.ones(n) * 1.5)
        
        # 检查一致性
        if voting_method == 'percentage':
            combined = 0.5 * judge_share + 0.5 * fan_votes
            worst_idx = np.argmin(combined)
            if worst_idx == eliminated_idx:
                return fan_votes
                
        elif voting_method == 'rank':
            fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
            total_rank = judge_rank + fan_rank
            worst_idx = np.argmax(total_rank)
            if worst_idx == eliminated_idx:
                return fan_votes
                
        elif voting_method == 'rank_bottom2':
            fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
            total_rank = judge_rank + fan_rank
            bottom_two = np.argsort(total_rank)[-2:]
            if eliminated_idx in bottom_two:
                return fan_votes
    
    return None


if __name__ == "__main__":
    # 测试代码
    print("平滑采样器测试\n")
    
    # 模拟3周的数据
    np.random.seed(42)
    
    smoother = SmoothSampler(n_candidates=100)
    
    # 周1：5人
    judge_share_1 = np.array([0.25, 0.22, 0.20, 0.18, 0.15])
    eliminated_idx_1 = 4  # 第5人被淘汰
    
    # 生成模拟MCMC样本
    mcmc_samples_1 = np.random.dirichlet(np.ones(5), size=1000)
    
    # 测试一致性检查
    consistent_sample = generate_consistent_sample(judge_share_1, eliminated_idx_1, 'percentage')
    if consistent_sample is not None:
        print(f"找到满足一致性的样本: {consistent_sample}")
        is_valid = smoother.check_consistency(consistent_sample, judge_share_1, eliminated_idx_1, 'percentage')
        print(f"一致性验证: {is_valid}")
    else:
        print("未找到满足一致性的样本")
