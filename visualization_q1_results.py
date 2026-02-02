import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from matplotlib import rcParams

# 设置中文字体（根据您的系统环境调整，如果乱码请改用英文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
# 设置绘图风格
sns.set_theme(style="whitegrid")

def load_data():
    """加载MCMC结果数据"""
    path = 'results/mcmc_smooth_results.csv'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    
    df = pd.read_csv(path)
    # 将字符串格式的列表转换为实际列表
    df['fan_votes_mean'] = df['fan_votes_mean'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['survivor_names'] = df['survivor_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def plot_season_heatmap(df, season_id, output_dir='figures'):
    """
    方案1：绘制指定赛季的粉丝支持率热力图
    """
    print(f"Drawing Heatmap for Season {season_id}...")
    season_df = df[df['season'] == season_id].sort_values('week')
    
    # 获取该赛季所有出现过的选手
    all_contestants = set()
    for names in season_df['survivor_names']:
        all_contestants.update(names)
    all_contestants = list(all_contestants)
    
    # 确定选手的排序（按最后存活的周数倒序，存活越久越靠上）
    last_week_presence = {}
    for c in all_contestants:
        presence = []
        for _, row in season_df.iterrows():
            if c in row['survivor_names']:
                presence.append(row['week'])
        last_week_presence[c] = max(presence) if presence else 0
    
    # 冠军在最上面
    sorted_contestants = sorted(all_contestants, key=lambda x: last_week_presence[x], reverse=True)
    
    # 构建矩阵: 行=选手, 列=周
    matrix_data = pd.DataFrame(index=sorted_contestants, columns=season_df['week'].unique())
    
    for _, row in season_df.iterrows():
        w = row['week']
        names = row['survivor_names']
        votes = row['fan_votes_mean']
        
        for name, vote in zip(names, votes):
            matrix_data.loc[name, w] = vote
            
    matrix_data = matrix_data.astype(float)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_data, cmap="YlOrRd", annot=True, fmt=".2f", 
                linewidths=.5, cbar_kws={'label': 'Estimated Fan Vote Share'})
    
    plt.title(f'Season {season_id} Fan Vote Evolution (MCMC Estimated)', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Contestant', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/S{season_id}_Heatmap.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.close()

def plot_voting_dynamics_multi(df, seasons, output_dir='figures'):
    """
    方案2：绘制多个赛季的前3名走势对比
    """
    print(f"Drawing Dynamics for Seasons {seasons}...")
    
    fig, axes = plt.subplots(len(seasons), 1, figsize=(10, 5 * len(seasons)))
    if len(seasons) == 1: axes = [axes]
    
    for idx, season_id in enumerate(seasons):
        ax = axes[idx]
        season_df = df[df['season'] == season_id].sort_values('week')
        
        # 找出最后一周的前4名
        last_week = season_df['week'].max()
        final_row = season_df[season_df['week'] == last_week].iloc[0]
        top_names = final_row['survivor_names'][:4] # 假设最后只剩3-4人
        
        # 收集这些人的历史数据
        history = {name: {'weeks': [], 'votes': []} for name in top_names}
        
        for _, row in season_df.iterrows():
            w = row['week']
            names = row['survivor_names']
            votes = row['fan_votes_mean']
            
            for n in top_names:
                if n in names:
                    v_idx = names.index(n)
                    history[n]['weeks'].append(w)
                    history[n]['votes'].append(votes[v_idx])
        
        # 绘图
        for name, data in history.items():
            ax.plot(data['weeks'], data['votes'], marker='o', linewidth=2, label=name)
            
        ax.set_title(f'Season {season_id} Top Contestants Trajectory', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Fan Vote Share')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = f"{output_dir}/Voting_Dynamics_Contrast.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.close()

def plot_elimination_scatter(df, output_dir='figures'):
    """
    方案3：评委分 vs 粉丝分 散点图 (区分 淘汰 vs 晋级)
    """
    print("Drawing Elimination Scatter Plot...")
    
    # 需要重新加载原始数据来计算评委份额
    raw_df = pd.read_csv('engineered_data.csv')
    
    scatter_data = []
    
    for _, row in df.iterrows():
        s = row['season']
        w = row['week']
        names = row['survivor_names']
        fan_votes = row['fan_votes_mean']
        eliminated = row['eliminated']
        
        # 计算该周评委份额
        # 简化逻辑：直接从engineered_data获取
        season_raw = raw_df[raw_df['season'] == s]
        judge_cols = [c for c in raw_df.columns if c.startswith(f'week{w}_judge') and c.endswith('_score')]
        
        # 获取该周每个人的评委分
        j_scores = []
        valid_indices = [] # 记录有评委分的对应names索引
        
        for i, name in enumerate(names):
             # 找到该人的行
            person_row = season_raw[season_raw['celebrity_name'] == name]
            if person_row.empty: 
                j_scores.append(0)
                continue
            
            score_sum = 0
            count = 0
            for col in judge_cols:
                val = person_row[col].values[0]
                if pd.notna(val):
                    score_sum += val
                    count += 1
            j_scores.append(score_sum)
            
        total_j = sum(j_scores)
        if total_j == 0: continue
        
        judge_shares = [x/total_j for x in j_scores]
        
        # 记录数据点
        for i, name in enumerate(names):
            status = 'Eliminated' if name == eliminated else 'Safe'
            # 过滤掉决赛周（Winner/Runner-up 不算普通意义的Safe/Eliminated逻辑，此处主要看中途淘汰机制）
            if row['n_survivors'] <= 3: 
                continue
            
            scatter_data.append({
                'season': s,
                'week': w,
                'name': name,
                'judge_share': judge_shares[i],
                'fan_share': fan_votes[i],
                'status': status
            })
            
    plot_df = pd.DataFrame(scatter_data)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制两个类别
    safe = plot_df[plot_df['status'] == 'Safe']
    elim = plot_df[plot_df['status'] == 'Eliminated']
    
    plt.scatter(safe['judge_share'], safe['fan_share'], 
                alpha=0.3, c='blue', label='Safe', s=30)
    plt.scatter(elim['judge_share'], elim['fan_share'], 
                alpha=0.8, c='red', marker='x', label='Eliminated', s=60)
    
    plt.title('Judge Score Share vs Fan Vote Share (Red=Eliminated)', fontsize=16)
    plt.xlabel('Judge Score Share (normalized)', fontsize=12)
    plt.ylabel('Estimated Fan Vote Share (normalized)', fontsize=12)
    plt.legend()
    
    # 画一条 y=x 线作为参考（虽然实际不完全是y=x决定）
    lims = [0, max(plot_df['judge_share'].max(), plot_df['fan_share'].max())]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='Equal Share Line')
    
    plt.tight_layout()
    save_path = f"{output_dir}/Judge_vs_Fan_Scatter.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    df = load_data()
    if df is not None:
        if not os.path.exists('figures'):
            os.makedirs('figures')
            
        # 1. 典型赛季热力图 (例如 Season 27 - Bobby Bones 赛季，或 Season 19)
        plot_season_heatmap(df, 27) # 争议最多的赛季
        # plot_season_heatmap(df, 19) # 另一个赛季
        
        # 2. 走势对比 (对比 S19 实力派获胜 vs S27 人气派获胜)
        plot_voting_dynamics_multi(df, [19, 27])
        
        # 3. 散点分布图
        plot_elimination_scatter(df)
        
        print("Done! Check 'figures' folder.")
