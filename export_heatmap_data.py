import pandas as pd
import ast
import os

def export_season_data_to_csv(season_id, input_path='results/mcmc_smooth_results.csv', output_path='S27_Heatmap_Data.csv'):
    """
    导出指定赛季的粉丝支持率数据为CSV格式
    格式: Name, Week1, Week2, ...
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 数据转换
    df['fan_votes_mean'] = df['fan_votes_mean'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['survivor_names'] = df['survivor_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # 筛选赛季
    season_df = df[df['season'] == season_id].sort_values('week')
    if season_df.empty:
        print(f"No data found for Season {season_id}")
        return

    # 获取该赛季所有出现过的选手
    all_contestants = set()
    for names in season_df['survivor_names']:
        all_contestants.update(names)
    all_contestants = list(all_contestants)
    
    # 确定选手的排序（按最后存活的周数倒序，存活越久越靠上，与热力图一致）
    last_week_presence = {}
    for c in all_contestants:
        presence = []
        for _, row in season_df.iterrows():
            if c in row['survivor_names']:
                presence.append(row['week'])
        last_week_presence[c] = max(presence) if presence else 0
    
    sorted_contestants = sorted(all_contestants, key=lambda x: last_week_presence[x], reverse=True)
    
    # 获取唯一的周数并排序
    weeks = sorted(season_df['week'].unique())
    col_names = [f"week{w}" for w in weeks]
    
    # 构建矩阵: 行=选手, 列=周
    matrix_data = pd.DataFrame(index=sorted_contestants, columns=col_names)
    matrix_data.index.name = 'Contestant Name'
    
    for _, row in season_df.iterrows():
        w = row['week']
        col_name = f"week{w}"
        names = row['survivor_names']
        votes = row['fan_votes_mean']
        
        for name, vote in zip(names, votes):
            matrix_data.loc[name, col_name] = vote
            
    # 填充NaN为0 (可选，根据需求，或者留空)
    # matrix_data = matrix_data.fillna(0)
    
    print(f"Exporting data to {output_path}...")
    matrix_data.to_csv(output_path)
    print("Done!")

if __name__ == '__main__':
    # 导出Season 27的数据（对应您提供的热力图）
    export_season_data_to_csv(27)
