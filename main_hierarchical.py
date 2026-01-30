"""
层次贝叶斯模型主程序
使用基本盘+表现票分解估算粉丝投票
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']

from src.data_loader import DWTSDataLoader
from src.hierarchical_sampler import HierarchicalMCMCSampler, SeasonData, prepare_season_data
from src.diagnostics import ModelDiagnostics, CertaintyMetrics


class HierarchicalVotingEstimator:
    """层次贝叶斯粉丝投票估算器"""
    
    def __init__(self, data_path: str = "engineered_data.csv",
                 output_dir: str = "results_hierarchical"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.loader = DWTSDataLoader(data_path)
        self.processed_data = None
        self.estimation_results = {}
        
        # 层次采样器
        self.sampler = HierarchicalMCMCSampler(
            n_iterations=15000,
            burn_in=5000,
            thinning=5,
            base_weight=0.7,  # 70%基本盘
            smoothness_prior_std=0.15
        )
    
    def load_data(self, seasons: List[int] = None):
        """加载数据"""
        print("=" * 60)
        print("步骤 1: 数据加载")
        print("=" * 60)
        
        self.loader.load_data()
        self.processed_data = self.loader.process_all_seasons(seasons)
        
        print(f"\n✓ 成功处理 {len(self.processed_data)} 个赛季")
    
    def estimate_season(self, season: int) -> Dict:
        """
        使用层次模型估算单个赛季
        
        Args:
            season: 赛季编号
            
        Returns:
            估算结果
        """
        print(f"\n{'=' * 60}")
        print(f"估算赛季 {season} (层次贝叶斯模型)")
        print(f"{'=' * 60}")
        
        # 准备数据
        season_data = prepare_season_data(self.processed_data, season)
        
        print(f"选手数: {season_data.n_contestants}")
        print(f"周数: {len(season_data.weeks)}")
        print(f"投票机制: {season_data.voting_method}")
        print(f"基本盘权重: {self.sampler.base_weight:.0%}")
        
        # 运行多链
        n_chains = 3
        all_alpha_samples = []
        all_results = []
        
        for chain_id in range(n_chains):
            print(f"  运行链 {chain_id + 1}/{n_chains}...", end=" ")
            result = self.sampler.sample_season(season_data)
            all_alpha_samples.append(result['alpha_samples'])
            all_results.append(result)
            print(f"接受率: {result['acceptance_rate']:.1%}")
        
        # 合并样本
        combined_alpha = np.vstack(all_alpha_samples)
        
        # Gelman-Rubin诊断
        r_hat = ModelDiagnostics.gelman_rubin_statistic(all_alpha_samples)
        converged = np.all(r_hat < 1.1)
        
        # 整理结果
        final_result = {
            'season': season,
            'n_contestants': season_data.n_contestants,
            'contestant_names': season_data.contestant_names,
            'voting_method': season_data.voting_method,
            'base_weight': self.sampler.base_weight,
            
            # 基本盘估计
            'alpha_mean': np.mean(combined_alpha, axis=0),
            'alpha_std': np.std(combined_alpha, axis=0),
            'alpha_samples': combined_alpha,
            
            # 每周粉丝投票
            'weekly_fan_votes': {},
            
            # 诊断
            'r_hat': r_hat,
            'converged': converged,
            'avg_acceptance_rate': np.mean([r['acceptance_rate'] for r in all_results])
        }
        
        # 合并每周投票
        for w in all_results[0]['fan_votes_by_week'].keys():
            weekly_samples = []
            for r in all_results:
                if w in r['fan_votes_by_week']:
                    weekly_samples.append(r['fan_votes_by_week'][w]['samples'])
            if weekly_samples:
                combined = np.vstack(weekly_samples)
                final_result['weekly_fan_votes'][w] = {
                    'mean': np.mean(combined, axis=0),
                    'std': np.std(combined, axis=0),
                    'samples': combined
                }
        
        # 计算一致性
        consistency = self._check_consistency(final_result, season_data)
        final_result['consistency_score'] = consistency
        
        print(f"\n结果:")
        print(f"  收敛: {'✓' if converged else '✗'}")
        print(f"  一致性: {consistency:.1%}")
        print(f"  平均接受率: {final_result['avg_acceptance_rate']:.1%}")
        
        # 显示基本盘排名
        print(f"\n基本盘份额排名 (Top 5):")
        alpha_mean = final_result['alpha_mean']
        sorted_idx = np.argsort(-alpha_mean)
        for rank, idx in enumerate(sorted_idx[:5]):
            name = season_data.contestant_names[idx]
            share = alpha_mean[idx]
            std = final_result['alpha_std'][idx]
            print(f"  {rank+1}. {name}: {share:.1%} ± {std:.1%}")
        
        self.estimation_results[season] = final_result
        return final_result
    
    def _check_consistency(self, result: Dict, season_data: SeasonData) -> float:
        """检查预测一致性"""
        correct = 0
        total = 0
        
        for w, week_info in enumerate(season_data.weeks):
            if week_info.get('is_no_elimination', False):
                continue
            
            if w not in result['weekly_fan_votes']:
                continue
            
            fan_votes_mean = result['weekly_fan_votes'][w]['mean']
            judge_share = week_info['judge_share']
            eliminated_indices = week_info.get('eliminated_indices_in_survivors', [])
            
            for elim_idx in eliminated_indices:
                if elim_idx is None or elim_idx >= len(fan_votes_mean):
                    continue
                
                # 预测淘汰
                if season_data.voting_method == 'percentage':
                    combined = 0.5 * judge_share + 0.5 * fan_votes_mean
                    predicted = np.argmin(combined)
                else:
                    n = len(fan_votes_mean)
                    jr = n + 1 - np.argsort(np.argsort(judge_share)) - 1
                    fr = n + 1 - np.argsort(np.argsort(fan_votes_mean)) - 1
                    predicted = np.argmax(jr + fr)
                
                if predicted == elim_idx:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def estimate_all_seasons(self, seasons: List[int] = None):
        """估算所有赛季"""
        if seasons is None:
            seasons = sorted(self.processed_data.keys())
        
        print("\n" + "=" * 60)
        print("步骤 2: 层次贝叶斯MCMC估算")
        print("=" * 60)
        
        for season in seasons:
            try:
                self.estimate_season(season)
            except Exception as e:
                print(f"\n赛季 {season} 发生错误: {e}")
                import traceback
                traceback.print_exc()
        
        self.save_results()
    
    def save_results(self):
        """保存结果"""
        # 保存完整结果
        results_path = self.output_dir / "hierarchical_results.pkl"
        
        # 移除大样本数组以减小文件大小
        save_results = {}
        for season, result in self.estimation_results.items():
            save_result = {k: v for k, v in result.items() 
                          if k not in ['alpha_samples']}
            # 简化weekly_fan_votes
            if 'weekly_fan_votes' in save_result:
                for w in save_result['weekly_fan_votes']:
                    if 'samples' in save_result['weekly_fan_votes'][w]:
                        del save_result['weekly_fan_votes'][w]['samples']
            save_results[season] = save_result
        
        with open(results_path, 'wb') as f:
            pickle.dump(save_results, f)
        print(f"\n✓ 结果已保存到: {results_path}")
        
        # 保存JSON摘要
        summary = self._generate_summary()
        summary_path = self.output_dir / "hierarchical_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✓ 摘要已保存到: {summary_path}")
    
    def _generate_summary(self) -> Dict:
        """生成摘要"""
        summary = {
            'model': 'Hierarchical Bayesian (Base + Performance)',
            'base_weight': self.sampler.base_weight,
            'total_seasons': len(self.estimation_results),
            'seasons': {}
        }
        
        for season, result in self.estimation_results.items():
            # 基本盘排名
            alpha_mean = result['alpha_mean']
            sorted_idx = np.argsort(-alpha_mean)
            
            base_ranking = []
            for idx in sorted_idx:
                base_ranking.append({
                    'name': result['contestant_names'][idx],
                    'base_share': float(alpha_mean[idx]),
                    'std': float(result['alpha_std'][idx])
                })
            
            summary['seasons'][int(season)] = {
                'n_contestants': result['n_contestants'],
                'voting_method': result['voting_method'],
                'converged': bool(result['converged']),
                'consistency_score': float(result['consistency_score']),
                'avg_acceptance_rate': float(result['avg_acceptance_rate']),
                'base_ranking': base_ranking[:10]  # Top 10
            }
        
        return summary
    
    def visualize_season(self, season: int):
        """可视化赛季结果"""
        if season not in self.estimation_results:
            print(f"赛季 {season} 尚未估算")
            return
        
        result = self.estimation_results[season]
        vis_dir = self.output_dir / f"season_{season}_viz"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 基本盘份额柱状图
        self._plot_base_shares(result, vis_dir)
        
        # 2. 粉丝投票时间序列
        self._plot_voting_timeline(result, season, vis_dir)
        
        print(f"✓ 可视化已保存到: {vis_dir}")
    
    def _plot_base_shares(self, result: Dict, output_dir: Path):
        """绘制基本盘份额"""
        names = result['contestant_names']
        alpha_mean = result['alpha_mean']
        alpha_std = result['alpha_std']
        
        # 按份额排序
        sorted_idx = np.argsort(-alpha_mean)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(names))
        
        plt.barh(y_pos, alpha_mean[sorted_idx], xerr=alpha_std[sorted_idx],
                 alpha=0.7, color='steelblue', capsize=3)
        plt.yticks(y_pos, [names[i] for i in sorted_idx])
        plt.xlabel('基本盘份额', fontsize=12)
        plt.title(f"赛季 {result['season']} 基本盘份额估计\n(λ = {result['base_weight']:.0%})",
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'base_shares.png', dpi=300)
        plt.close()
    
    def _plot_voting_timeline(self, result: Dict, season: int, output_dir: Path):
        """绘制投票时间线"""
        weekly = result['weekly_fan_votes']
        if not weekly:
            return
        
        weeks = sorted(weekly.keys())
        n_weeks = len(weeks)
        
        # 获取每周的存活选手和投票
        # 简化：只显示前5名基本盘选手的时间序列
        alpha_mean = result['alpha_mean']
        top5_idx = np.argsort(-alpha_mean)[:5]
        names = result['contestant_names']
        
        plt.figure(figsize=(14, 8))
        
        for idx in top5_idx:
            name = names[idx]
            votes = []
            valid_weeks = []
            
            for w in weeks:
                week_votes = weekly[w]['mean']
                # 需要找到这个选手在这周的索引
                # 这里简化处理，假设选手顺序一致
                if idx < len(week_votes):
                    votes.append(week_votes[idx] if idx < len(week_votes) else np.nan)
                    valid_weeks.append(w + 1)
            
            if votes:
                plt.plot(valid_weeks, votes, marker='o', label=name, linewidth=2)
        
        plt.xlabel('周次', fontsize=12)
        plt.ylabel('粉丝投票份额', fontsize=12)
        plt.title(f"赛季 {season} 粉丝投票趋势 (Top 5 基本盘选手)", fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'voting_timeline.png', dpi=300)
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("DWTS 层次贝叶斯粉丝投票估算")
    print("模型: 基本盘 + 表现票 分解")
    print("=" * 60)
    
    estimator = HierarchicalVotingEstimator()
    
    try:
        # 加载数据 (测试前3个赛季)
        estimator.load_data(seasons=list(range(1, 4)))
        
        # 估算
        estimator.estimate_all_seasons()
        
        # 可视化
        for season in estimator.estimation_results.keys():
            estimator.visualize_season(season)
        
        print("\n" + "=" * 60)
        print("✓ 层次贝叶斯分析完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
