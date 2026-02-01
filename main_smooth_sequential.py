"""
MCMC采样主程序 - 思路i：顺序生成平滑数据
保证100%一致性的同时，最小化相邻周的变化

注意：Windows兼容版本，禁用多进程以避免pickle问题
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Windows多进程兼容性标志
USE_MULTIPROCESSING = os.name != 'nt'  # Windows上禁用多进程

if USE_MULTIPROCESSING:
    import concurrent.futures

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DWTSDataLoader
from mcmc_sampler import MCMCSampler
from smooth_sampler import generate_consistent_sample


def get_fan_base_prior(df: pd.DataFrame, season: int, survivor_names: list) -> np.ndarray:
    """
    根据粉丝基数模型计算第一周的先验概率
    
    Formula: Score = 1 + 2*B1 + 1*B2 + 3*(1 - PartnerRank)
    Args:
        df: 包含 engineered features 的原始数据
        season: 当前赛季
        survivor_names: 当前仍存活的选手列表（第一周通常是所有人）
    Returns:
        归一化的先验概率向量
    """
    season_df = df[df['season'] == season]
    scores = []
    
    for name in survivor_names:
        # Get contestant row
        row = season_df[season_df['celebrity_name'] == name]
        if row.empty:
            # Fallback if name mismatch (shouldn't happen with clean data)
            scores.append(1.0)
            continue
            
        row = row.iloc[0]
        
        # Extract features
        b1 = row.get('total_fan_saves_bottom1', 0)
        b2 = row.get('total_fan_saves_bottom2', 0)
        partner_rank = row.get('partner_avg_placement', 0.5) # Default to mid if missing
        
        # Compute Score
        # Note: partner_avg_placement is [0,1], lower is better (rank 1 is best).
        # We want high score for good partner (low rank percentile).
        score = 1.0 + (b1 * 2.0) + (b2 * 1.0) + (3.0 * (1.0 - partner_rank))
        scores.append(score)
        
    scores = np.array(scores)
    return scores / scores.sum() # Normalize


def check_consistency(fan_votes: np.ndarray, judge_share: np.ndarray,
                      eliminated_idx: int, voting_method: str,
                      judge_ranks: np.ndarray = None) -> bool:
    """检查粉丝投票是否满足一致性"""
    if voting_method == 'percentage':
        combined = 0.5 * judge_share + 0.5 * fan_votes
        worst_idx = np.argmin(combined)
        return worst_idx == eliminated_idx
    elif voting_method in ['rank', 'rank_bottom2']:
        n = len(fan_votes)
        if judge_ranks is not None:
            judge_rank = judge_ranks
        else:
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


def calculate_smoothness_distance(prev_votes: np.ndarray, curr_votes: np.ndarray,
                                  prev_names: list, curr_names: list) -> float:
    """
    计算两周投票分布之间的平滑距离
    只比较共同存活的选手
    
    Args:
        prev_votes: 上一周的投票份额
        curr_votes: 本周的投票份额
        prev_names: 上一周的选手名单
        curr_names: 本周的选手名单
        
    Returns:
        L2距离（归一化后）
    """
    # 找出共同的选手
    common_names = set(prev_names) & set(curr_names)
    
    if len(common_names) == 0:
        return float('inf')
    
    # 提取共同选手的投票份额
    prev_common = []
    curr_common = []
    
    for name in common_names:
        prev_idx = prev_names.index(name)
        curr_idx = curr_names.index(name)
        prev_common.append(prev_votes[prev_idx])
        curr_common.append(curr_votes[curr_idx])
    
    prev_common = np.array(prev_common)
    curr_common = np.array(curr_common)
    
    # 归一化（因为共同选手的份额和不一定为1）
    prev_norm = prev_common / (prev_common.sum() + 1e-10)
    curr_norm = curr_common / (curr_common.sum() + 1e-10)
    
    # L2距离
    return np.sqrt(np.sum((prev_norm - curr_norm) ** 2))


def select_smoothest_sample(valid_samples: np.ndarray,
                            prev_votes: np.ndarray,
                            prev_names: list,
                            curr_names: list,
                            prior_votes: np.ndarray = None) -> np.ndarray:
    """
    从有效样本中选择与上一周最平滑的一个
    
    Args:
        valid_samples: 满足一致性的样本集合
        prev_votes: 上一周的投票份额（None表示第一周）
        prev_names: 上一周的选手名单
        curr_names: 本周的选手名单
        prior_votes: 第一周的先验分布（可选）
        
    Returns:
        选中的样本
    """
    if prev_votes is None or len(valid_samples) == 1:
        # 第一周
        if prior_votes is not None:
             # 选择距离先验最近的样本
             distances = [np.sqrt(np.sum((s - prior_votes) ** 2)) for s in valid_samples]
             return valid_samples[np.argmin(distances)]
        
        # Fallback: 重心法 (如果没有先验)
        mean_sample = valid_samples.mean(axis=0)
        distances = [np.sqrt(np.sum((s - mean_sample) ** 2)) for s in valid_samples]
        return valid_samples[np.argmin(distances)]
    
    # 后续周：选择与上一周变化最小的
    min_dist = float('inf')
    best_sample = valid_samples[0]
    
    for sample in valid_samples:
        dist = calculate_smoothness_distance(prev_votes, sample, prev_names, curr_names)
        if dist < min_dist:
            min_dist = dist
            best_sample = sample
    
    return best_sample


def process_season_smooth(season: int, loader: DWTSDataLoader, 
                          sampler: MCMCSampler) -> dict:
    """
    处理单个赛季，使用顺序生成平滑方法
    
    思路i：
    - 第1周：从满足一致性的MCMC样本中，选择接近均值的
    - 第k周：从满足一致性的MCMC样本中，选择与第k-1周变化最小的
    """
    # 确保进程间的随机性不同，但结果可复现
    np.random.seed(season * 999)
    
    voting_method = loader.get_voting_method(season)
    season_processed = loader.process_season(season)
    
    # 获取最大周数
    max_week = season_processed['max_week']
    
    season_result = {
        'season': season,
        'voting_method': voting_method,
        'weekly_results': [],
        'total_weeks': 0,
        'consistent_weeks': 0
    }
    
    # 顺序处理每周，维护上一周的状态
    prev_votes = None
    prev_names = None
    
    for week in range(1, max_week + 1):
        if week not in season_processed['weeks']:
            continue
        
        week_data = season_processed['weeks'][week]
        
        # 跳过决赛周
        if week_data.get('is_finale', False):
            continue
        
        judge_share = week_data['judge_share']
        judge_ranks = week_data.get('judge_ranks', None)
        survivor_names = week_data['survivor_names']
        eliminated_indices = week_data.get('eliminated_indices_in_survivors', [])
        
        # 处理无淘汰周 (No Elimination Week)
        if not eliminated_indices:
            # 这种情况约束条件为空，可行解是整个单纯形
            # 根据平滑假设，我们应该保持上一周的投票分布不变（仅做归一化适配）
            # 或者如果是第一周，则使用先验
            
            # 计算本周的先验（可能用于Week 1或作为参考）
            prior = get_fan_base_prior(loader.raw_data, season, survivor_names)
            
            if prev_votes is None:
                # 第一周且无淘汰：直接使用粉丝基数先验
                selected_votes = prior
                method = 'prior_only (no elim)'
            else:
                # 后续周无淘汰：投影上一周的结果
                # 创建一个映射
                prev_map = {name: val for name, val in zip(prev_names, prev_votes)}
                
                # 提取本周选手的上周得票
                temp_votes = []
                for name in survivor_names:
                    if name in prev_map:
                        temp_votes.append(prev_map[name])
                    else:
                        # 如果是新加入的选手（极少见），给予平均值或先验值
                        temp_votes.append(1.0 / len(survivor_names))
                
                temp_votes = np.array(temp_votes)
                selected_votes = temp_votes / temp_votes.sum()
                method = 'smooth_projection (no elim)'
            
            # 记录无淘汰周结果
            week_result = {
                'week': week,
                'eliminated': "None (No Elimination)",
                'n_survivors': len(survivor_names),
                'survivor_names': survivor_names,
                'fan_votes': selected_votes,
                'is_consistent': True, # 无约束，天然满足
                'method': method,
                'acceptance_rate': 1.0
            }
            season_result['weekly_results'].append(week_result)
            season_result['total_weeks'] += 1
            season_result['consistent_weeks'] += 1
            
            # 更新状态
            prev_votes = selected_votes
            prev_names = survivor_names
            continue

        # 处理正常淘汰周
        for elim_idx in eliminated_indices:
            if elim_idx >= len(survivor_names):
                continue
            
            eliminated_name = survivor_names[elim_idx]
            
            # 运行MCMC采样
            try:
                mcmc_samples = sampler.sample_week(judge_share, elim_idx, voting_method, judge_ranks)
            except:
                mcmc_samples = None
            
            # 筛选满足一致性的样本
            valid_samples = []
            if mcmc_samples is not None:
                for s in mcmc_samples:
                    if check_consistency(s, judge_share, elim_idx, voting_method, judge_ranks):
                        valid_samples.append(s)
            
            if len(valid_samples) > 0:
                valid_samples = np.array(valid_samples)
                
                # 计算先验（仅第一周）
                prior = None
                if prev_votes is None:
                    prior = get_fan_base_prior(loader.raw_data, season, survivor_names)

                # 使用平滑选择策略
                selected_votes = select_smoothest_sample(
                    valid_samples, prev_votes, prev_names, survivor_names, prior_votes=prior
                )
                method = 'mcmc_smooth' if prev_votes is not None else 'mcmc_prior'
                is_consistent = True
            else:
                # 使用拒绝采样
                selected_votes = generate_consistent_sample(
                    judge_share, elim_idx, voting_method, 
                    judge_ranks=judge_ranks, n_attempts=50000
                )
                if selected_votes is not None:
                    method = 'rejection_sampling'
                    is_consistent = True
                else:
                    selected_votes = np.ones(len(judge_share)) / len(judge_share)
                    method = 'failed'
                    is_consistent = False
            
            # 记录结果
            week_result = {
                'week': week,
                'eliminated': eliminated_name,
                'n_survivors': len(survivor_names),
                'survivor_names': survivor_names,
                'fan_votes': selected_votes,
                'is_consistent': is_consistent,
                'method': method,
                'acceptance_rate': sampler.acceptance_rate if mcmc_samples is not None else 0.0
            }
            
            season_result['weekly_results'].append(week_result)
            season_result['total_weeks'] += 1
            if is_consistent:
                season_result['consistent_weeks'] += 1
            
            # 更新上一周状态（用于下一周的平滑）
            prev_votes = selected_votes
            prev_names = survivor_names
    
    # 计算一致性率
    if season_result['total_weeks'] > 0:
        season_result['consistency_rate'] = season_result['consistent_weeks'] / season_result['total_weeks']
    else:
        season_result['consistency_rate'] = 0.0
    
    return season_result


def main():
    print("=" * 70)
    print("MCMC采样器 - 思路i：顺序生成平滑数据")
    print("=" * 70)
    
    # 加载数据
    data_path = Path(__file__).parent / "engineered_data.csv"
    loader = DWTSDataLoader(str(data_path))
    loader.load_data()
    
    # MCMC设置 - Windows上减少迭代次数以避免内存问题
    n_iterations = 50000 if os.name == 'nt' else 100000
    sampler = MCMCSampler(n_iterations=n_iterations, burn_in=5000, thinning=10, proposal_sigma=0.3)
    
    seasons = sorted(loader.raw_data['season'].unique())
    all_results = []
    
    mode = "顺序模式" if os.name == 'nt' else "并行模式"
    print(f"处理 {len(seasons)} 个赛季 ({mode})...")
    print(f"MCMC设置: n={sampler.n_iterations}, burn={sampler.burn_in}, thin={sampler.thinning}")
    if os.name == 'nt':
        print("注意: Windows系统，已禁用多进程以保证稳定性")
    print("-" * 70)
    
    method_counts = {}
    start_time = time.time()
    
    # 根据操作系统选择处理方式
    if USE_MULTIPROCESSING:
        # macOS/Linux: 使用多进程并行处理
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 提交所有任务
            future_to_season = {
                executor.submit(process_season_smooth, season, loader, sampler): season 
                for season in seasons
            }
            
            # 获取结果（按照完成顺序）
            for i, future in enumerate(concurrent.futures.as_completed(future_to_season)):
                season = future_to_season[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # 统计方法使用次数
                    for wr in result['weekly_results']:
                        method = wr['method']
                        method_counts[method] = method_counts.get(method, 0) + 1
                    
                    elapsed = time.time() - start_time
                    print(f"[{i+1}/{len(seasons)}] 赛季 {season:2d}: {result['voting_method']:15s} | "
                          f"一致性={result['consistency_rate']:.1%} | "
                          f"耗时={elapsed:.1f}s")
                          
                except Exception as e:
                    print(f"赛季 {season} 处理出错: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Windows: 顺序处理，避免multiprocessing问题
        for i, season in enumerate(seasons):
            try:
                result = process_season_smooth(season, loader, sampler)
                all_results.append(result)
                
                # 统计方法使用次数
                for wr in result['weekly_results']:
                    method = wr['method']
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                elapsed = time.time() - start_time
                print(f"[{i+1}/{len(seasons)}] 赛季 {season:2d}: {result['voting_method']:15s} | "
                      f"一致性={result['consistency_rate']:.1%} | "
                      f"耗时={elapsed:.1f}s")
                      
            except Exception as e:
                print(f"赛季 {season} 处理出错: {e}")
                import traceback
                traceback.print_exc()

    # 重新按赛季排序
    all_results.sort(key=lambda x: x['season'])
    
    # 汇总统计
    print()
    print("=" * 70)
    print("汇总统计（思路i：顺序生成平滑）")
    print("=" * 70)
    
    total_weeks = sum(r['total_weeks'] for r in all_results)
    consistent_weeks = sum(r['consistent_weeks'] for r in all_results)
    
    print(f"总处理周数: {total_weeks}")
    print(f"一致周数: {consistent_weeks}")
    
    if total_weeks > 0:
        print(f"总体一致性: {consistent_weeks/total_weeks:.2%}")
    else:
        print("总体一致性: N/A (无有效周数)")
        print("错误: 未能处理任何周数据，请检查数据文件和依赖项")
        return all_results
    
    print()
    print("各方法使用次数:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")
    
    # 保存结果
    output_path = Path(__file__).parent / "results" / "mcmc_smooth_results.csv"
    
    rows = []
    for result in all_results:
        for wr in result['weekly_results']:
            rows.append({
                'season': result['season'],
                'voting_method': result['voting_method'],
                'week': wr['week'],
                'eliminated': wr['eliminated'],
                'n_survivors': wr['n_survivors'],
                'is_consistent': wr['is_consistent'],
                'acceptance_rate': wr['acceptance_rate'],
                'method': wr['method'],
                'fan_votes_mean': str(wr['fan_votes'].tolist()),
                'survivor_names': str(wr['survivor_names'])
            })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 计算平滑程度
    print()
    print("=" * 70)
    print("平滑程度评估")
    print("=" * 70)
    
    all_changes = []
    
    for result in all_results:
        prev_votes = None
        prev_names = None
        
        for wr in result['weekly_results']:
            curr_votes = wr['fan_votes']
            curr_names = wr['survivor_names']
            
            if prev_votes is not None:
                dist = calculate_smoothness_distance(prev_votes, curr_votes, prev_names, curr_names)
                if dist < float('inf'):
                    all_changes.append({
                        'season': result['season'],
                        'week': wr['week'],
                        'change': dist,
                        'n_survivors': wr['n_survivors']
                    })
            
            prev_votes = curr_votes
            prev_names = curr_names
    
    changes_df = pd.DataFrame(all_changes)
    print(f"相邻周变化统计（归一化L2距离）:")
    print(f"  样本数:   {len(changes_df)}")
    print(f"  平均变化: {changes_df['change'].mean():.4f}")
    print(f"  标准差:   {changes_df['change'].std():.4f}")
    print(f"  最大变化: {changes_df['change'].max():.4f}")
    print(f"  最小变化: {changes_df['change'].min():.4f}")
    
    print()
    print("变化最大的10周:")
    print(changes_df.nlargest(10, 'change')[['season', 'week', 'change', 'n_survivors']].to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    results = main()
