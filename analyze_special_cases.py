"""分析特殊淘汰情况"""
import pandas as pd
import re
from collections import Counter

df = pd.read_csv('engineered_data.csv')

print('=== 特殊淘汰情况分析 ===')

for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]
    
    # 提取淘汰周
    elim_info = []
    finalists = []
    for _, row in season_df.iterrows():
        result = str(row['results'])
        name = row['celebrity_name']
        placement = row['placement']
        
        if 'Week' in result:
            match = re.search(r'Week (\d+)', result)
            if match:
                elim_info.append((name, int(match.group(1)), placement))
        else:
            # 决赛选手 (1st, 2nd, 3rd Place等)
            finalists.append((name, placement, result))
    
    # 统计每周淘汰人数
    elim_weeks = [e[1] for e in elim_info]
    week_counts = Counter(elim_weeks)
    
    # 找出特殊情况
    multi_elim = {w: c for w, c in week_counts.items() if c > 1}
    
    # 找出没有淘汰的周
    max_week = max(elim_weeks) if elim_weeks else 0
    all_weeks = set(range(1, max_week + 1))
    elim_week_set = set(elim_weeks)
    no_elim_weeks = all_weeks - elim_week_set
    
    if multi_elim or no_elim_weeks or len(finalists) > 3:
        print(f'\n赛季 {season}:')
        if multi_elim:
            print(f'  多人淘汰周: {multi_elim}')
            for name, week, place in elim_info:
                if week in multi_elim:
                    print(f'    - {name} (Week {week}, placement={place})')
        if no_elim_weeks:
            print(f'  无人淘汰周: {sorted(no_elim_weeks)}')
        if len(finalists) > 3:
            print(f'  决赛人数: {len(finalists)}')
            for name, place, result in sorted(finalists, key=lambda x: x[1]):
                print(f'    - {name}: {result} (placement={place})')
