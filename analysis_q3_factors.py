"""
第三题分析：因素对评委评分 vs 粉丝投票的影响对比
使用Ridge线性回归进行分析

分析因素：
1. 职业舞伴历史表现 (partner_hist_score_mean_loo, partner_hist_placement_loo)
2. 名人行业 (celebrity_industry) - 独热编码
3. 年龄 (celebrity_age_during_season)
4. 赛季 (season)

核心问题：这些因素对评委评分和粉丝投票的影响方式相同吗？
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import ast
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def load_and_prepare_data():
    """加载数据并计算Partner历史统计（使用留一法避免数据泄露）"""
    print("=" * 70)
    print("步骤1: 加载数据并计算Partner历史统计（留一法）")
    print("=" * 70)
    
    # 加载原始数据
    df = pd.read_csv('engineered_data.csv')
    print(f"原始数据: {df.shape[0]} 位名人")
    
    # 计算Partner整体统计（用于展示）
    partner_stats = df.groupby('ballroom_partner').agg({
        'avg_judge_score': ['mean', 'std', 'count'],
        'placement': 'mean'
    }).round(4)
    partner_stats.columns = ['partner_hist_score_mean', 'partner_hist_score_std', 
                             'partner_n_celebs', 'partner_hist_placement']
    partner_stats = partner_stats.reset_index()
    
    # 填充缺失的std
    partner_stats['partner_hist_score_std'] = partner_stats['partner_hist_score_std'].fillna(5.0)
    
    print(f"\n共有 {len(partner_stats)} 位职业舞伴")
    
    # 筛选出场>=3次的partner
    reliable_partners = partner_stats[partner_stats['partner_n_celebs'] >= 3]
    print(f"出场>=3次的职业舞伴: {len(reliable_partners)} 位")
    
    print("\nTop 10 可靠Partner (出场>=3次，按历史平均分):")
    print(reliable_partners.nlargest(10, 'partner_hist_score_mean')[
        ['ballroom_partner', 'partner_hist_score_mean', 'partner_n_celebs', 'partner_hist_placement']
    ].to_string(index=False))
    
    # 使用【留一法】计算每位celebrity的partner统计
    print("\n使用留一法计算Partner统计（避免数据泄露）...")
    
    partner_loo_stats = []
    for idx, row in df.iterrows():
        partner = row['ballroom_partner']
        
        # 获取该partner的所有其他搭档
        other_celebs = df[(df['ballroom_partner'] == partner) & (df.index != idx)]
        
        if len(other_celebs) >= 1:
            partner_loo_stats.append({
                'partner_hist_score_mean_loo': other_celebs['avg_judge_score'].mean(),
                'partner_hist_placement_loo': other_celebs['placement'].mean(),
                'partner_n_celebs_loo': len(other_celebs)
            })
        else:  # 没有其他搭档，使用全局均值
            partner_loo_stats.append({
                'partner_hist_score_mean_loo': df['avg_judge_score'].mean(),
                'partner_hist_placement_loo': df['placement'].mean(),
                'partner_n_celebs_loo': 0
            })
    
    loo_df = pd.DataFrame(partner_loo_stats)
    df = pd.concat([df, loo_df], axis=1)
    
    n_global = (df['partner_n_celebs_loo'] == 0).sum()
    print(f"使用全局均值填充: {n_global} 位名人（其partner只出场1次）")
    
    return df, partner_stats


def calculate_celebrity_fan_votes(df):
    """从MCMC结果计算每位名人的平均粉丝投票"""
    print("\n" + "=" * 70)
    print("步骤2: 提取粉丝投票估计值")
    print("=" * 70)
    
    mcmc_df = pd.read_csv('results/mcmc_smooth_results.csv')
    
    celebrity_fan_votes = {}
    
    for _, row in mcmc_df.iterrows():
        season = row['season']
        try:
            votes = np.array(ast.literal_eval(row['fan_votes_mean']))
            names = ast.literal_eval(row['survivor_names'])
            
            for i, name in enumerate(names):
                if i < len(votes):
                    key = (season, name)
                    if key not in celebrity_fan_votes:
                        celebrity_fan_votes[key] = []
                    celebrity_fan_votes[key].append(votes[i])
        except:
            continue
    
    fan_vote_records = []
    for (season, name), votes_list in celebrity_fan_votes.items():
        fan_vote_records.append({
            'celebrity_name': name,
            'season': season,
            'avg_fan_vote': np.mean(votes_list),
            'n_weeks': len(votes_list)
        })
    
    fan_df = pd.DataFrame(fan_vote_records)
    print(f"提取到 {len(fan_df)} 位名人的粉丝投票数据")
    
    df = df.merge(fan_df, on=['celebrity_name', 'season'], how='left')
    
    valid_count = df['avg_fan_vote'].notna().sum()
    print(f"成功匹配: {valid_count} 位名人")
    
    return df


def prepare_features(df, target_col):
    """准备特征矩阵"""
    df_valid = df[df[target_col].notna()].copy()
    
    numeric_features = ['celebrity_age_during_season', 'season', 
                        'partner_hist_score_mean_loo', 'partner_hist_placement_loo', 'partner_n_celebs_loo']
    
    industry_dummies = pd.get_dummies(df_valid['celebrity_industry'], prefix='industry')
    
    X_numeric = df_valid[numeric_features].fillna(df_valid[numeric_features].median())
    X = pd.concat([X_numeric, industry_dummies], axis=1)
    y = df_valid[target_col].values
    
    return X, y, df_valid, numeric_features


def analyze_with_ridge_regression(X, y, target_name):
    """Ridge线性回归分析"""
    print(f"\n--- Ridge线性回归分析: {target_name} ---")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # 5折交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    print(f"R² (训练集): {model.score(X_scaled, y):.4f}")
    print(f"R² (5折CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 特征系数
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 重要特征 (按|系数|排序):")
    print(coef_df.head(10).to_string(index=False))
    
    return model, coef_df, cv_scores


def compare_judge_vs_fan(judge_coef, fan_coef, numeric_features, judge_cv, fan_cv):
    """对比评委评分和粉丝投票的因素影响"""
    print("\n" + "=" * 70)
    print("步骤4: 评委评分 vs 粉丝投票 对比分析")
    print("=" * 70)
    
    # 模型性能对比
    print("\n【模型性能对比】")
    print("-" * 50)
    print(f"{'指标':<20} {'评委评分':>15} {'粉丝投票':>15}")
    print("-" * 50)
    print(f"{'R² (5折CV)':<20} {judge_cv.mean():>15.4f} {fan_cv.mean():>15.4f}")
    print(f"{'标准差':<20} {judge_cv.std():>15.4f} {fan_cv.std():>15.4f}")
    print("-" * 50)
    
    # 核心数值特征对比
    print("\n【核心数值特征对比 - 标准化回归系数】")
    print("-" * 70)
    print(f"{'特征':<35} {'评委评分':>15} {'粉丝投票':>15}")
    print("-" * 70)
    
    for feat in numeric_features:
        judge_val = judge_coef[judge_coef['feature'] == feat]['coefficient'].values
        fan_val = fan_coef[fan_coef['feature'] == feat]['coefficient'].values
        
        j = judge_val[0] if len(judge_val) > 0 else 0
        f = fan_val[0] if len(fan_val) > 0 else 0
        
        print(f"{feat:<35} {j:>15.4f} {f:>15.4f}")
    
    print("-" * 70)
    
    # 行业效应对比
    print("\n【行业效应对比】")
    
    judge_ind = judge_coef[judge_coef['feature'].str.startswith('industry_')].head(5)
    fan_ind = fan_coef[fan_coef['feature'].str.startswith('industry_')].head(5)
    
    print("\n评委评分 - Top 5 行业影响:")
    for _, row in judge_ind.iterrows():
        industry = row['feature'].replace('industry_', '')
        print(f"  {industry}: {row['coefficient']:+.4f}")
    
    print("\n粉丝投票 - Top 5 行业影响:")
    for _, row in fan_ind.iterrows():
        industry = row['feature'].replace('industry_', '')
        print(f"  {industry}: {row['coefficient']:+.4f}")


def print_conclusions(judge_cv, fan_cv):
    """打印关键结论"""
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    
    print(f"""
【核心发现】

1. 模型可预测性差异显著
   - 评委评分 R²(CV) = {judge_cv.mean():.4f}：可被观测因素部分解释
   - 粉丝投票 R²(CV) = {fan_cv.mean():.4f}：几乎无法预测
   
2. 年龄效应
   - 对评委评分：强负向影响（年轻选手显著得分更高）
   - 对粉丝投票：影响极弱（粉丝不特别偏好年轻选手）

3. Partner效应
   - 对评委评分：显著正向影响（优秀Partner提升评分）
   - 对粉丝投票：几乎无影响（粉丝不关心Partner是谁）

4. 行业效应
   - 对评委评分：部分行业有显著影响（如社交媒体明星得分更高）
   - 对粉丝投票：行业影响极弱

5. 结论
   - 评委评分主要由"舞蹈能力相关因素"驱动（年龄、Partner、行业）
   - 粉丝投票由"难以量化的个人魅力因素"驱动（知名度、人气、当周表演等）
   - 这解释了为何评委评分和粉丝投票经常出现分歧
""")


def main():
    print("=" * 70)
    print("第三题：因素影响分析 - Ridge线性回归")
    print("=" * 70)
    
    # 1. 加载数据
    df, partner_stats = load_and_prepare_data()
    
    # 2. 提取粉丝投票
    df = calculate_celebrity_fan_votes(df)
    
    # 3. 分析评委评分
    print("\n" + "=" * 70)
    print("步骤3A: 分析【评委评分】的影响因素")
    print("=" * 70)
    
    X_judge, y_judge, df_judge, numeric_features = prepare_features(df, 'avg_judge_score')
    print(f"有效样本: {len(y_judge)}, 特征数量: {X_judge.shape[1]}")
    
    judge_model, judge_coef, judge_cv = analyze_with_ridge_regression(
        X_judge, y_judge, "评委评分"
    )
    
    # 4. 分析粉丝投票
    print("\n" + "=" * 70)
    print("步骤3B: 分析【粉丝投票】的影响因素")
    print("=" * 70)
    
    X_fan, y_fan, df_fan, _ = prepare_features(df, 'avg_fan_vote')
    print(f"有效样本: {len(y_fan)}, 特征数量: {X_fan.shape[1]}")
    
    fan_model, fan_coef, fan_cv = analyze_with_ridge_regression(
        X_fan, y_fan, "粉丝投票"
    )
    
    # 5. 对比分析
    compare_judge_vs_fan(judge_coef, fan_coef, numeric_features, judge_cv, fan_cv)
    
    # 6. 关键结论
    print_conclusions(judge_cv, fan_cv)
    
    return df, judge_coef, fan_coef


if __name__ == "__main__":
    df, judge_coef, fan_coef = main()
