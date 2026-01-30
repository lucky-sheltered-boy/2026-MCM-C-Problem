"""
主程序：DWTS粉丝投票估算系统
整合数据加载、MCMC采样和模型诊断
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List
import warnings
import sys

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 导入自定义模块
from data_loader import DWTSDataLoader
from mcmc_sampler import MCMCSampler
from diagnostics import ModelDiagnostics, CertaintyMetrics, ConsistencyValidator


class DWTSVotingEstimator:
    """DWTS粉丝投票估算主类"""
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        初始化估算器
        
        Args:
            data_path: 预处理后的数据文件路径 (engineered_data.csv)
            output_dir: 输出结果目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.loader = DWTSDataLoader(data_path)
        self.sampler = MCMCSampler(
            n_iterations=10000,
            burn_in=2000,
            thinning=5,
            proposal_sigma=0.3
        )
        self.validator = ConsistencyValidator()
        
        # 存储结果
        self.processed_data = None
        self.estimation_results = {}
        
    def load_and_process_data(self, seasons: List[int] = None):
        """加载和处理数据"""
        print("=" * 60)
        print("步骤 1: 数据加载")
        print("=" * 60)
        
        self.loader.load_data()
        self.processed_data = self.loader.process_all_seasons(seasons)
        
        print(f"\n✓ 成功处理 {len(self.processed_data)} 个赛季")
        
    def estimate_fan_votes_single_week(self, season: int, week: int,
                                       n_chains: int = 3) -> Dict:
        """
        估算单周的粉丝投票（运行多条链）
        
        支持：
        - 单人淘汰（正常情况）
        - 多人淘汰
        - 无人淘汰（仅记录，不估算）
        - 决赛周（估算最终排名）
        
        Args:
            season: 赛季编号
            week: 周次
            n_chains: MCMC链数量
            
        Returns:
            估算结果字典
        """
        season_data = self.processed_data[season]
        week_data = season_data['weeks'][week]
        
        # 获取评委得分份额
        judge_share = week_data['judge_share']
        survivors = week_data['survivors']
        voting_method = season_data['voting_method']
        
        # 处理特殊情况
        is_no_elimination = week_data['is_no_elimination']
        is_multi_elimination = week_data['is_multi_elimination']
        is_finale = week_data['is_finale']
        n_eliminated = week_data['n_eliminated']
        
        # 无人淘汰周：仅记录，不估算
        if is_no_elimination and not is_finale:
            return {
                'season': season,
                'week': week,
                'voting_method': voting_method,
                'n_survivors': len(survivors),
                'is_no_elimination': True,
                'is_finale': False,
                'survivor_names': week_data['survivor_names'],
                'judge_share': judge_share,
                'note': '本周无人淘汰'
            }
        
        # 决赛周：估算最终排名的粉丝投票
        if is_finale:
            return self._estimate_finale_week(season, week, n_chains)
        
        # 正常淘汰周（单人或多人）
        eliminated_indices = week_data['eliminated_indices_in_survivors']
        
        if len(eliminated_indices) == 0:
            return None
        
        # 对于多人淘汰，我们逐个估算每个被淘汰者
        # 假设淘汰是按某种顺序进行的（这是一个简化假设）
        results_list = []
        
        for elim_idx in eliminated_indices:
            # 运行多条MCMC链
            chains = []
            acceptance_rates = []
            
            for chain_id in range(n_chains):
                samples = self.sampler.sample_week(
                    judge_share, elim_idx, voting_method
                )
                chains.append(samples)
                acceptance_rates.append(self.sampler.acceptance_rate)
            
            # 合并所有链的样本
            all_samples = np.vstack(chains)
            
            # 计算诊断指标
            r_hat = ModelDiagnostics.gelman_rubin_statistic(chains)
            ess = ModelDiagnostics.effective_sample_size(all_samples)
            hpdi = self.sampler.calculate_hpdi(all_samples)
            hpdi_widths = CertaintyMetrics.calculate_hpdi_width(all_samples)
            posterior_std = CertaintyMetrics.calculate_posterior_std(all_samples)
            cv = CertaintyMetrics.calculate_coefficient_of_variation(all_samples)
            
            # 后验预测检查
            ppc_score = ModelDiagnostics.posterior_predictive_check(
                all_samples, judge_share, elim_idx, voting_method
            )
            
            eliminated_name = week_data['survivor_names'][elim_idx]
            
            result = {
                'eliminated_celebrity': eliminated_name,
                'eliminated_idx_in_survivors': elim_idx,
                'fan_votes_mean': np.mean(all_samples, axis=0),
                'fan_votes_std': posterior_std,
                'fan_votes_hpdi': hpdi,
                'samples': all_samples,
                'r_hat': r_hat,
                'ess': ess,
                'acceptance_rates': acceptance_rates,
                'avg_acceptance_rate': np.mean(acceptance_rates),
                'hpdi_widths': hpdi_widths,
                'coefficient_of_variation': cv,
                'consistency_score': ppc_score,
                'converged': np.all(r_hat < 1.1)
            }
            results_list.append(result)
        
        # 整理最终结果
        final_result = {
            'season': season,
            'week': week,
            'voting_method': voting_method,
            'n_survivors': len(survivors),
            'survivor_names': week_data['survivor_names'],
            'judge_share': judge_share,
            'is_no_elimination': False,
            'is_multi_elimination': is_multi_elimination,
            'is_finale': False,
            'n_eliminated': n_eliminated,
            'eliminations': results_list,  # 每个被淘汰者的详细结果
        }
        
        # 计算整体一致性（所有淘汰者的平均）
        if results_list:
            final_result['avg_consistency_score'] = np.mean([r['consistency_score'] for r in results_list])
            final_result['all_converged'] = all(r['converged'] for r in results_list)
            final_result['avg_acceptance_rate'] = np.mean([r['avg_acceptance_rate'] for r in results_list])
            # 兼容性：如果只有一人被淘汰，保留旧字段
            if len(results_list) == 1:
                r = results_list[0]
                final_result['eliminated_celebrity'] = r['eliminated_celebrity']
                final_result['fan_votes_mean'] = r['fan_votes_mean']
                final_result['fan_votes_std'] = r['fan_votes_std']
                final_result['fan_votes_hpdi'] = r['fan_votes_hpdi']
                final_result['samples'] = r['samples']
                final_result['r_hat'] = r['r_hat']
                final_result['ess'] = r['ess']
                final_result['hpdi_widths'] = r['hpdi_widths']
                final_result['consistency_score'] = r['consistency_score']
                final_result['converged'] = r['converged']
        
        return final_result
    
    def _estimate_finale_week(self, season: int, week: int, n_chains: int = 3) -> Dict:
        """估算决赛周的粉丝投票（决定最终排名）"""
        season_data = self.processed_data[season]
        week_data = season_data['weeks'][week]
        
        judge_share = week_data['judge_share']
        survivors = week_data['survivors']
        voting_method = season_data['voting_method']
        finale_rankings = week_data.get('finale_rankings', [])
        
        # 运行MCMC采样
        chains = []
        acceptance_rates = []
        
        for chain_id in range(n_chains):
            # 决赛使用特殊的似然函数（需要匹配最终排名）
            samples = self.sampler.sample_week_finale(
                judge_share, finale_rankings, voting_method
            )
            chains.append(samples)
            acceptance_rates.append(self.sampler.acceptance_rate)
        
        all_samples = np.vstack(chains)
        
        # 诊断
        r_hat = ModelDiagnostics.gelman_rubin_statistic(chains)
        hpdi = self.sampler.calculate_hpdi(all_samples)
        hpdi_widths = CertaintyMetrics.calculate_hpdi_width(all_samples)
        posterior_std = CertaintyMetrics.calculate_posterior_std(all_samples)
        
        # 验证排名一致性
        ranking_consistency = self._check_finale_ranking_consistency(
            all_samples, judge_share, finale_rankings, voting_method
        )
        
        return {
            'season': season,
            'week': week,
            'voting_method': voting_method,
            'n_survivors': len(survivors),
            'survivor_names': week_data['survivor_names'],
            'judge_share': judge_share,
            'is_no_elimination': False,
            'is_finale': True,
            'finale_rankings': finale_rankings,
            'fan_votes_mean': np.mean(all_samples, axis=0),
            'fan_votes_std': posterior_std,
            'fan_votes_hpdi': hpdi,
            'hpdi_widths': hpdi_widths,
            'samples': all_samples,
            'r_hat': r_hat,
            'converged': np.all(r_hat < 1.1),
            'avg_acceptance_rate': np.mean(acceptance_rates),
            'ranking_consistency': ranking_consistency
        }
    
    def _check_finale_ranking_consistency(self, samples: np.ndarray, 
                                          judge_share: np.ndarray,
                                          finale_rankings: List[Dict],
                                          voting_method: str) -> float:
        """检查决赛排名的一致性"""
        if not finale_rankings:
            return 0.0
        
        correct_count = 0
        n_samples = samples.shape[0]
        
        for fan_votes in samples:
            # 计算综合得分并排序
            if voting_method in ['rank', 'rank_bottom2']:
                n = len(fan_votes)
                judge_rank = n + 1 - np.argsort(np.argsort(judge_share)) - 1
                fan_rank = n + 1 - np.argsort(np.argsort(fan_votes)) - 1
                combined = judge_rank + fan_rank
                predicted_order = np.argsort(combined)  # 从好到差
            else:  # percentage
                combined = 0.5 * judge_share + 0.5 * fan_votes
                predicted_order = np.argsort(-combined)  # 从高到低
            
            # 检查预测顺序是否与实际排名匹配
            actual_order = [f['survivor_idx'] for f in finale_rankings]
            if list(predicted_order[:len(actual_order)]) == actual_order:
                correct_count += 1
        
        return correct_count / n_samples
    
    def estimate_season(self, season: int, n_chains: int = 3):
        """估算整个赛季的粉丝投票（处理所有特殊情况）"""
        print(f"\n{'=' * 60}")
        print(f"估算赛季 {season}")
        print(f"{'=' * 60}")
        
        season_data = self.processed_data[season]
        season_results = {}
        
        for week_num in sorted(season_data['weeks'].keys()):
            week_data = season_data['weeks'][week_num]
            
            print(f"\n周次 {week_num}: ", end="")
            
            result = self.estimate_fan_votes_single_week(season, week_num, n_chains)
            
            if result is None:
                print("跳过（无有效数据）")
                continue
            
            season_results[week_num] = result
            
            # 根据不同情况显示信息
            if result.get('is_no_elimination') and not result.get('is_finale'):
                print(f"无人淘汰（存活 {result['n_survivors']} 人）")
            elif result.get('is_finale'):
                n_finalists = len(result.get('finale_rankings', []))
                consistency = result.get('ranking_consistency', 0)
                converged = result.get('converged', False)
                print(f"决赛周 ({n_finalists} 人决赛)")
                print(f"  收敛: {'✓' if converged else '✗'} | "
                      f"排名一致性: {consistency:.1%} | "
                      f"接受率: {result['avg_acceptance_rate']:.1%}")
                # 显示决赛排名
                for f in result.get('finale_rankings', []):
                    print(f"    第{f['final_place']}名: {f['name']}")
            elif result.get('is_multi_elimination'):
                print(f"多人淘汰 ({result['n_eliminated']} 人):")
                for elim in result.get('eliminations', []):
                    print(f"    - {elim['eliminated_celebrity']}: "
                          f"一致性 {elim['consistency_score']:.1%}")
                print(f"  整体收敛: {'✓' if result.get('all_converged') else '✗'} | "
                      f"平均一致性: {result.get('avg_consistency_score', 0):.1%}")
            else:
                # 单人淘汰（正常情况）
                print(f"淘汰 {result.get('eliminated_celebrity', 'Unknown')}")
                print(f"  收敛: {'✓' if result.get('converged') else '✗'} | "
                      f"一致性: {result.get('consistency_score', 0):.1%} | "
                      f"接受率: {result.get('avg_acceptance_rate', 0):.1%}")
        
        self.estimation_results[season] = season_results
        return season_results
    
    def estimate_all_seasons(self, seasons: List[int] = None):
        """估算所有赛季"""
        print("\n" + "=" * 60)
        print("步骤 2: MCMC采样估算粉丝投票")
        print("=" * 60)
        
        if seasons is None:
            seasons = sorted(self.processed_data.keys())
        
        for season in seasons:
            self.estimate_season(season)
        
        # 保存估算结果
        self.save_results()
    
    def save_results(self):
        """保存估算结果"""
        results_path = self.output_dir / "estimation_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.estimation_results, f)
        print(f"\n✓ 结果已保存到: {results_path}")
        
        # 生成JSON摘要（不含大数组）
        summary = self.generate_summary()
        summary_path = self.output_dir / "results_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ 摘要已保存到: {summary_path}")
    
    def generate_summary(self) -> Dict:
        """生成结果摘要（处理特殊情况）"""
        summary = {
            'total_seasons': len(self.estimation_results),
            'seasons': {}
        }
        
        for season, season_results in self.estimation_results.items():
            # 过滤出有采样结果的周次（跳过零淘汰周）
            estimation_weeks = {k: v for k, v in season_results.items() 
                               if not v.get('is_no_elimination', False)}
            
            # 计算收敛周数（处理不同字段名）
            def is_converged(r):
                if r.get('is_multi_elimination'):
                    return r.get('all_converged', False)
                return r.get('converged', False)
            
            # 计算一致性得分（处理不同字段名）
            def get_consistency(r):
                if r.get('is_finale'):
                    return r.get('ranking_consistency', 0)
                if r.get('is_multi_elimination'):
                    return r.get('avg_consistency_score', 0)
                return r.get('consistency_score', 0)
            
            season_summary = {
                'total_weeks': len(season_results),
                'estimation_weeks': len(estimation_weeks),
                'no_elimination_weeks': len(season_results) - len(estimation_weeks),
                'converged_weeks': sum(1 for r in estimation_weeks.values() if is_converged(r)),
                'avg_consistency': np.mean([get_consistency(r) for r in estimation_weeks.values()]) if estimation_weeks else 0,
                'avg_acceptance_rate': np.mean([r.get('avg_acceptance_rate', 0) 
                                               for r in estimation_weeks.values()]) if estimation_weeks else 0,
                'weeks': {}
            }
            
            for week, result in season_results.items():
                if result.get('is_no_elimination', False):
                    week_summary = {
                        'type': 'no_elimination',
                        'n_survivors': result['n_survivors']
                    }
                elif result.get('is_finale', False):
                    week_summary = {
                        'type': 'finale',
                        'n_finalists': result['n_survivors'],
                        'finale_rankings': [f['name'] for f in result['finale_rankings']],
                        'ranking_consistency': float(result['ranking_consistency']),
                        'converged': bool(result['converged']),
                        'fan_votes_mean': result['fan_votes_mean'].tolist(),
                        'fan_votes_std': result['fan_votes_std'].tolist(),
                        'hpdi_widths': result['hpdi_widths'].tolist()
                    }
                elif result.get('is_multi_elimination', False):
                    # 多人淘汰周：汇总各被淘汰者的结果
                    eliminations_summary = []
                    for elim in result.get('eliminations', []):
                        eliminations_summary.append({
                            'name': elim['eliminated_celebrity'],
                            'consistency': float(elim['consistency_score']),
                            'converged': bool(elim['converged'])
                        })
                    week_summary = {
                        'type': 'multi_elimination',
                        'n_eliminated': result['n_eliminated'],
                        'eliminations': eliminations_summary,
                        'avg_consistency': float(result.get('avg_consistency_score', 0)),
                        'all_converged': bool(result.get('all_converged', False))
                    }
                else:
                    # 单人淘汰（正常情况）
                    week_summary = {
                        'type': 'elimination',
                        'eliminated': result.get('eliminated_celebrity', 'Unknown'),
                        'consistency': float(result.get('consistency_score', 0)),
                        'converged': bool(result.get('converged', False)),
                        'fan_votes_mean': result['fan_votes_mean'].tolist(),
                        'fan_votes_std': result['fan_votes_std'].tolist(),
                        'hpdi_widths': result['hpdi_widths'].tolist()
                    }
                season_summary['weeks'][week] = week_summary
            
            summary['seasons'][season] = season_summary
        
        return summary
    
    def visualize_season(self, season: int):
        """可视化赛季结果"""
        print(f"\n{'=' * 60}")
        print(f"步骤 3: 可视化赛季 {season} 结果")
        print(f"{'=' * 60}")
        
        season_results = self.estimation_results.get(season)
        if season_results is None:
            print(f"赛季 {season} 尚未估算")
            return
        
        # 创建输出目录
        vis_dir = self.output_dir / f"season_{season}_visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 一致性趋势图
        self._plot_consistency_trend(season, season_results, vis_dir)
        
        # 2. 确定性度量图
        self._plot_certainty_metrics(season, season_results, vis_dir)
        
        # 3. 示例周次的后验分布 (选择第一个有样本的周次)
        valid_weeks = {w: r for w, r in season_results.items() 
                       if not r.get('is_no_elimination', False) and 'samples' in r}
        if valid_weeks:
            example_week = sorted(valid_weeks.keys())[0]
            self._plot_posterior_distributions(season, example_week, valid_weeks[example_week], vis_dir)
        
        print(f"✓ 可视化结果已保存到: {vis_dir}")
    
    def _plot_consistency_trend(self, season: int, season_results: Dict, output_dir: Path):
        """绘制一致性趋势"""
        # 过滤掉零淘汰周
        valid_weeks = {w: r for w, r in season_results.items() 
                       if not r.get('is_no_elimination', False)}
        
        if not valid_weeks:
            print("  无有效周次用于绘制一致性趋势")
            return
        
        weeks = sorted(valid_weeks.keys())
        # 处理决赛周和普通淘汰周不同的字段名
        consistency = [valid_weeks[w].get('consistency_score', 
                       valid_weeks[w].get('ranking_consistency', 0)) for w in weeks]
        
        plt.figure(figsize=(10, 6))
        plt.plot(weeks, consistency, marker='o', linewidth=2, markersize=8)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
        plt.xlabel('周次', fontsize=12)
        plt.ylabel('一致性分数', fontsize=12)
        plt.title(f'赛季 {season} 模型预测一致性趋势', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'consistency_trend.png', dpi=300)
        plt.close()
    
    def _plot_certainty_metrics(self, season: int, season_results: Dict, output_dir: Path):
        """绘制确定性度量"""
        # 过滤掉零淘汰周，并确保有hpdi_widths数据
        valid_weeks = {}
        for w, r in season_results.items():
            if r.get('is_no_elimination', False):
                continue
            # 处理多人淘汰周（数据在eliminations列表中）
            if 'hpdi_widths' in r:
                valid_weeks[w] = r['hpdi_widths']
            elif 'eliminations' in r and r['eliminations']:
                # 取第一个淘汰者的hpdi_widths
                valid_weeks[w] = r['eliminations'][0].get('hpdi_widths', None)
        
        # 过滤掉None值
        valid_weeks = {w: v for w, v in valid_weeks.items() if v is not None}
        
        if not valid_weeks:
            print("  无有效周次用于绘制确定性度量")
            return
        
        weeks = sorted(valid_weeks.keys())
        
        # 计算平均HPDI宽度
        avg_hpdi_widths = [np.mean(valid_weeks[w]) for w in weeks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(weeks, avg_hpdi_widths, alpha=0.7, color='steelblue')
        ax.set_xlabel('周次', fontsize=12)
        ax.set_ylabel('平均HPDI宽度', fontsize=12)
        ax.set_title(f'赛季 {season} 估算不确定性（HPDI宽度）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'certainty_metrics.png', dpi=300)
        plt.close()
    
    def _plot_posterior_distributions(self, season: int, week: int, result: Dict, output_dir: Path):
        """绘制后验分布"""
        samples = result['samples']
        n_contestants = samples.shape[1]
        
        fig, axes = plt.subplots(1, n_contestants, figsize=(4*n_contestants, 4))
        if n_contestants == 1:
            axes = [axes]
        
        for i in range(n_contestants):
            axes[i].hist(samples[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].axvline(result['fan_votes_mean'][i], color='red', linestyle='--', 
                          linewidth=2, label=f"均值: {result['fan_votes_mean'][i]:.3f}")
            axes[i].axvline(result['fan_votes_hpdi'][i, 0], color='orange', linestyle=':',
                          label=f"95% HPDI")
            axes[i].axvline(result['fan_votes_hpdi'][i, 1], color='orange', linestyle=':')
            axes[i].set_xlabel('粉丝投票份额', fontsize=10)
            axes[i].set_ylabel('频数', fontsize=10)
            axes[i].set_title(f'选手 {i+1}', fontsize=11, fontweight='bold')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'赛季 {season} 周次 {week} 粉丝投票后验分布', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'posterior_week_{week}.png', dpi=300)
        plt.close()


def main():
    """主函数"""
    # 设置中文字体（用于matplotlib）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 初始化估算器 - 使用队友预处理后的数据
    estimator = DWTSVotingEstimator(
        data_path="engineered_data.csv",
        output_dir="results"
    )
    
    # 运行分析流程
    try:
        # 步骤1: 数据加载（全部34个赛季）
        all_seasons = list(range(1, 35))  # 赛季1-34
        estimator.load_and_process_data(seasons=all_seasons)
        
        # 步骤2: MCMC采样估算
        estimator.estimate_all_seasons()
        
        # 步骤3: 可视化结果
        for season in all_seasons:
            estimator.visualize_season(season)
        
        print("\n" + "=" * 60)
        print("✓ 所有分析完成！")
        print("=" * 60)
        
        # 打印摘要
        summary = estimator.generate_summary()
        print("\n分析摘要:")
        for season, info in summary['seasons'].items():
            print(f"\n赛季 {season}:")
            print(f"  总周数: {info['total_weeks']}")
            print(f"  收敛周数: {info['converged_weeks']}")
            print(f"  平均一致性: {info['avg_consistency']:.2%}")
            print(f"  平均接受率: {info['avg_acceptance_rate']:.2%}")
        
    except FileNotFoundError:
        print("\n错误: 未找到数据文件 'engineered_data.csv'")
        print("请确保 2026-MCM-C-Problem/engineered_data.csv 存在。")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
