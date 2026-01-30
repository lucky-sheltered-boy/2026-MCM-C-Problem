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
        
        Args:
            season: 赛季编号
            week: 周次
            n_chains: MCMC链数量
            
        Returns:
            估算结果字典
        """
        season_data = self.processed_data[season]
        week_data = season_data['weeks'][week]
        
        # 获取评委得分份额（已在data_loader中计算）
        judge_share = week_data['judge_share']
        survivors = week_data['survivors']
        
        # 找到被淘汰选手在存活者中的索引
        eliminated_idx_in_survivors = week_data['eliminated_idx_in_survivors']
        if eliminated_idx_in_survivors is None:
            return None
        
        # 获取投票方法
        voting_method = season_data['voting_method']
        
        # 运行多条MCMC链
        chains = []
        acceptance_rates = []
        
        for chain_id in range(n_chains):
            samples = self.sampler.sample_week(
                judge_share, eliminated_idx_in_survivors, voting_method
            )
            chains.append(samples)
            acceptance_rates.append(self.sampler.acceptance_rate)
        
        # 合并所有链的样本
        all_samples = np.vstack(chains)
        
        # 计算Gelman-Rubin统计量
        r_hat = ModelDiagnostics.gelman_rubin_statistic(chains)
        
        # 计算有效样本量
        ess = ModelDiagnostics.effective_sample_size(all_samples)
        
        # 计算确定性度量
        hpdi = self.sampler.calculate_hpdi(all_samples)
        hpdi_widths = CertaintyMetrics.calculate_hpdi_width(all_samples)
        posterior_std = CertaintyMetrics.calculate_posterior_std(all_samples)
        cv = CertaintyMetrics.calculate_coefficient_of_variation(all_samples)
        
        # 后验预测检查
        ppc_score = ModelDiagnostics.posterior_predictive_check(
            all_samples, judge_share, eliminated_idx_in_survivors, voting_method
        )
        
        # 整理结果
        eliminated_idx = week_data['eliminated_idx']
        eliminated_name = week_data['survivor_names'][eliminated_idx_in_survivors] if eliminated_idx_in_survivors is not None else "Unknown"
        
        result = {
            'season': season,
            'week': week,
            'voting_method': voting_method,
            'n_survivors': len(survivors),
            'eliminated_celebrity': eliminated_name,
            
            # 后验估计
            'fan_votes_mean': np.mean(all_samples, axis=0),
            'fan_votes_std': posterior_std,
            'fan_votes_hpdi': hpdi,
            
            # 评委信息
            'judge_share': judge_share,
            
            # 样本
            'samples': all_samples,
            
            # 诊断指标
            'r_hat': r_hat,
            'ess': ess,
            'acceptance_rates': acceptance_rates,
            'avg_acceptance_rate': np.mean(acceptance_rates),
            
            # 确定性度量
            'hpdi_widths': hpdi_widths,
            'coefficient_of_variation': cv,
            
            # 一致性
            'consistency_score': ppc_score,
            
            # 收敛状态
            'converged': np.all(r_hat < 1.1)
        }
        
        return result
    
    def estimate_season(self, season: int, n_chains: int = 3):
        """估算整个赛季的粉丝投票"""
        print(f"\n{'=' * 60}")
        print(f"估算赛季 {season}")
        print(f"{'=' * 60}")
        
        season_data = self.processed_data[season]
        season_results = {}
        
        for week_num in sorted(season_data['weeks'].keys()):
            week_data = season_data['weeks'][week_num]
            
            # 跳过没有淘汰的周
            if week_data['eliminated_idx_in_survivors'] is None:
                continue
            
            print(f"\n周次 {week_num}: ", end="")
            
            result = self.estimate_fan_votes_single_week(season, week_num, n_chains)
            
            if result is not None:
                season_results[week_num] = result
                
                # 显示简要信息
                print(f"淘汰 {result['eliminated_celebrity']}")
                print(f"  收敛: {'✓' if result['converged'] else '✗'} | "
                      f"一致性: {result['consistency_score']:.1%} | "
                      f"接受率: {result['avg_acceptance_rate']:.1%}")
        
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
        """生成结果摘要"""
        summary = {
            'total_seasons': len(self.estimation_results),
            'seasons': {}
        }
        
        for season, season_results in self.estimation_results.items():
            season_summary = {
                'total_weeks': len(season_results),
                'converged_weeks': sum(1 for r in season_results.values() if r['converged']),
                'avg_consistency': np.mean([r['consistency_score'] for r in season_results.values()]),
                'avg_acceptance_rate': np.mean([r['avg_acceptance_rate'] for r in season_results.values()]),
                'weeks': {}
            }
            
            for week, result in season_results.items():
                week_summary = {
                    'eliminated': result['eliminated_celebrity'],
                    'consistency': float(result['consistency_score']),
                    'converged': bool(result['converged']),
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
        
        # 3. 示例周次的后验分布
        example_week = list(season_results.keys())[0]
        self._plot_posterior_distributions(season, example_week, season_results[example_week], vis_dir)
        
        print(f"✓ 可视化结果已保存到: {vis_dir}")
    
    def _plot_consistency_trend(self, season: int, season_results: Dict, output_dir: Path):
        """绘制一致性趋势"""
        weeks = sorted(season_results.keys())
        consistency = [season_results[w]['consistency_score'] for w in weeks]
        
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
        weeks = sorted(season_results.keys())
        
        # 计算平均HPDI宽度
        avg_hpdi_widths = [np.mean(season_results[w]['hpdi_widths']) for w in weeks]
        
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
        # 步骤1: 数据加载（测试前3个赛季）
        estimator.load_and_process_data(seasons=[1, 2, 3])
        
        # 步骤2: MCMC采样估算
        estimator.estimate_all_seasons()
        
        # 步骤3: 可视化结果
        for season in [1, 2, 3]:
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
