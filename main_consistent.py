"""
MCMC采样主程序（带100%一致性保证）
在原有main.py基础上增加一致性强制保证

核心改进：
1. Acceptance rate只统计burn-in后的数据
2. 通过拒绝采样或筛选MCMC样本，保证100%一致性
3. 可选的平滑采样（顺序生成各周数据）
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DWTSDataLoader
from mcmc_sampler import MCMCSampler
from smooth_sampler import SmoothSampler, generate_consistent_sample


def check_consistency(fan_votes: np.ndarray, judge_share: np.ndarray,
                      eliminated_idx: int, voting_method: str) -> bool:
    """检查粉丝投票是否满足一致性"""
    if voting_method == 'percentage':
        combined = 0.5 * judge_share + 0.5 * fan_votes
        worst_idx = np.argmin(combined)
        return worst_idx == eliminated_idx
    elif voting_method in ['rank', 'rank_bottom2']:
        n = len(fan_votes)
        judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
        fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
        total_rank = judge_rank + fan_rank
        if voting_method == 'rank':
            worst_idx = np.argmax(total_rank)
            return worst_idx == eliminated_idx
        else:
            bottom_two = np.argsort(total_rank)[-2:]
            return eliminated_idx in bottom_two
    return False


def estimate_week_with_consistency(sampler: MCMCSampler, 
                                   judge_share: np.ndarray,
                                   eliminated_idx: int,
                                   voting_method: str,
                                   ensure_100_percent: bool = True) -> dict:
    """
    估算单周粉丝投票，可选强制100%一致性
    
    Args:
        sampler: MCMC采样器
        judge_share: 评委得分份额
        eliminated_idx: 被淘汰选手索引
        voting_method: 投票方法
        ensure_100_percent: 是否强制100%一致性
        
    Returns:
        包含估算结果的字典
    """
    # 先进行MCMC采样
    try:
        mcmc_samples = sampler.sample_week(judge_share, eliminated_idx, voting_method)
        acceptance_rate = sampler.acceptance_rate
    except Exception as e:
        mcmc_samples = None
        acceptance_rate = 0.0
    
    if not ensure_100_percent:
        # 不强制一致性，直接返回MCMC结果
        if mcmc_samples is not None and len(mcmc_samples) > 0:
            fan_mean = mcmc_samples.mean(axis=0)
            is_consistent = check_consistency(fan_mean, judge_share, eliminated_idx, voting_method)
            return {
                'fan_votes_mean': fan_mean,
                'fan_votes_std': mcmc_samples.std(axis=0),
                'is_consistent': is_consistent,
                'acceptance_rate': acceptance_rate,
                'n_samples': len(mcmc_samples),
                'method': 'mcmc'
            }
        else:
            return {
                'fan_votes_mean': np.ones(len(judge_share)) / len(judge_share),
                'fan_votes_std': np.zeros(len(judge_share)),
                'is_consistent': False,
                'acceptance_rate': 0.0,
                'n_samples': 0,
                'method': 'failed'
            }
    
    # 强制100%一致性
    if mcmc_samples is not None and len(mcmc_samples) > 0:
        # 从MCMC样本中筛选满足一致性的
        valid_samples = []
        for s in mcmc_samples:
            if check_consistency(s, judge_share, eliminated_idx, voting_method):
                valid_samples.append(s)
        
        if len(valid_samples) > 0:
            valid_samples = np.array(valid_samples)
            return {
                'fan_votes_mean': valid_samples.mean(axis=0),
                'fan_votes_std': valid_samples.std(axis=0),
                'is_consistent': True,
                'acceptance_rate': acceptance_rate,
                'n_samples': len(valid_samples),
                'n_valid_ratio': len(valid_samples) / len(mcmc_samples),
                'method': 'mcmc_filtered'
            }
    
    # MCMC样本都不满足约束，使用拒绝采样
    fan_votes = generate_consistent_sample(
        judge_share, eliminated_idx, voting_method, n_attempts=50000
    )
    
    if fan_votes is not None:
        return {
            'fan_votes_mean': fan_votes,
            'fan_votes_std': np.zeros(len(judge_share)),  # 单个样本无法估计标准差
            'is_consistent': True,
            'acceptance_rate': acceptance_rate,
            'n_samples': 1,
            'method': 'rejection_sampling'
        }
    else:
        # 实在找不到，返回失败
        return {
            'fan_votes_mean': np.ones(len(judge_share)) / len(judge_share),
            'fan_votes_std': np.zeros(len(judge_share)),
            'is_consistent': False,
            'acceptance_rate': 0.0,
            'n_samples': 0,
            'method': 'failed_no_valid_sample'
        }


def process_all_seasons(data_path: str, ensure_consistency: bool = True) -> dict:
    """
    处理所有赛季
    
    Args:
        data_path: 数据文件路径
        ensure_consistency: 是否强制100%一致性
        
    Returns:
        所有赛季的结果
    """
    # 加载数据
    loader = DWTSDataLoader(data_path)
    loader.load_data()
    
    sampler = MCMCSampler(n_iterations=10000, burn_in=2000, thinning=5, proposal_sigma=0.3)
    
    seasons = sorted(loader.raw_data['season'].unique())
    all_results = []
    
    print(f"处理 {len(seasons)} 个赛季...")
    print("-" * 70)
    
    for season in seasons:
        voting_method = loader.get_voting_method(season)
        season_df = loader.raw_data[loader.raw_data['season'] == season].copy()
        
        # 获取最大周数
        score_cols = [c for c in season_df.columns if c.startswith('week') and 'percent_score' in c]
        max_week = max([int(c.split('_')[0].replace('week', '')) for c in score_cols])
        
        # 获取淘汰信息
        season_processed = loader.process_season(season)
        elimination_map = season_processed['elimination_map']  # week -> [indices]
        
        season_result = {
            'season': season,
            'voting_method': voting_method,
            'weekly_results': [],
            'total_weeks': 0,
            'consistent_weeks': 0
        }
        
        # 处理每周
        for week in range(1, max_week + 1):
            if week not in season_processed['weeks']:
                continue
            
            week_data = season_processed['weeks'][week]
            
            # 跳过无淘汰周
            if week_data.get('is_no_elimination', False) and not week_data.get('is_finale', False):
                continue
            
            # 跳过决赛周（单独处理）
            if week_data.get('is_finale', False):
                continue
            
            judge_share = week_data['judge_share']
            survivor_names = week_data['survivor_names']
            eliminated_indices = week_data.get('eliminated_indices_in_survivors', [])
            
            for elim_idx in eliminated_indices:
                if elim_idx >= len(survivor_names):
                    continue
                
                eliminated_name = survivor_names[elim_idx]
                
                # 估算粉丝投票
                result = estimate_week_with_consistency(
                    sampler, judge_share, elim_idx, voting_method, 
                    ensure_100_percent=ensure_consistency
                )
                
                week_result = {
                    'week': week,
                    'eliminated': eliminated_name,
                    'n_survivors': len(survivor_names),
                    **result
                }
                
                season_result['weekly_results'].append(week_result)
                season_result['total_weeks'] += 1
                if result['is_consistent']:
                    season_result['consistent_weeks'] += 1
        
        # 计算一致性率
        if season_result['total_weeks'] > 0:
            season_result['consistency_rate'] = season_result['consistent_weeks'] / season_result['total_weeks']
        else:
            season_result['consistency_rate'] = 0.0
        
        all_results.append(season_result)
        
        print(f"赛季 {season:2d}: {voting_method:15s} | "
              f"周数={season_result['total_weeks']:2d} | "
              f"一致性={season_result['consistency_rate']:.1%}")
    
    return all_results


def main():
    print("=" * 70)
    print("MCMC采样器 - 带100%一致性保证")
    print("=" * 70)
    
    data_path = Path(__file__).parent / "engineered_data.csv"
    if not data_path.exists():
        print(f"错误：找不到数据文件 {data_path}")
        return
    
    # 运行两种模式对比
    print("\n【模式1: 强制100%一致性】")
    results_consistent = process_all_seasons(str(data_path), ensure_consistency=True)
    
    # 汇总
    print("\n" + "=" * 70)
    print("汇总统计（100%一致性模式）")
    print("=" * 70)
    
    total_weeks = sum(r['total_weeks'] for r in results_consistent)
    total_consistent = sum(r['consistent_weeks'] for r in results_consistent)
    overall_rate = total_consistent / total_weeks if total_weeks > 0 else 0.0
    
    print(f"总处理周数: {total_weeks}")
    print(f"一致周数: {total_consistent}")
    print(f"总体一致性: {overall_rate:.2%}")
    
    # 统计使用的方法
    methods_count = {}
    for r in results_consistent:
        for wr in r['weekly_results']:
            m = wr.get('method', 'unknown')
            methods_count[m] = methods_count.get(m, 0) + 1
    
    print("\n各方法使用次数:")
    for m, c in sorted(methods_count.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")
    
    # 保存结果
    output_path = Path(__file__).parent / "results" / "mcmc_consistent_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    
    rows = []
    for season_result in results_consistent:
        for wr in season_result['weekly_results']:
            rows.append({
                'season': season_result['season'],
                'voting_method': season_result['voting_method'],
                'week': wr['week'],
                'eliminated': wr['eliminated'],
                'n_survivors': wr['n_survivors'],
                'is_consistent': wr['is_consistent'],
                'acceptance_rate': wr['acceptance_rate'],
                'method': wr.get('method', ''),
                'n_samples': wr.get('n_samples', 0),
                'fan_votes_mean': str(wr['fan_votes_mean'].tolist())
            })
    
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
    
    return results_consistent


if __name__ == "__main__":
    results = main()
