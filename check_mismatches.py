"""检查排名法赛季的不一致周"""
import pickle
import numpy as np

with open('results/estimation_results.pkl', 'rb') as f:
    r = pickle.load(f)

# 检查排名法赛季的不一致周
rank_seasons = list(range(1, 3)) + list(range(28, 35))
mismatches = []

for season in rank_seasons:
    if season not in r:
        continue
    for week in r[season].keys():
        week_data = r[season][week]
        if week_data.get('is_no_elimination') or week_data.get('is_finale'):
            continue
        
        survivor_names = week_data.get('survivor_names', [])
        judge_share = np.array(week_data.get('judge_share', []))
        fan_share = np.array(week_data.get('fan_votes_mean', []))
        actual = week_data.get('eliminated_celebrity')
        
        if len(survivor_names) < 2 or actual is None:
            continue
        
        # 计算排名法淘汰
        n = len(survivor_names)
        j_order = np.argsort(-judge_share)
        j_ranks = np.empty(n, dtype=int)
        j_ranks[j_order] = np.arange(1, n+1)
        
        f_order = np.argsort(-fan_share)
        f_ranks = np.empty(n, dtype=int)
        f_ranks[f_order] = np.arange(1, n+1)
        
        total_ranks = j_ranks + f_ranks
        elim_idx = np.argmax(total_ranks)
        predicted = survivor_names[elim_idx]
        
        if predicted != actual:
            mismatches.append({
                'season': season,
                'week': week,
                'predicted': predicted,
                'actual': actual,
                'note': week_data.get('note', '')
            })

print(f'Mismatches in Rank seasons: {len(mismatches)}')
for m in mismatches:
    print(f"  S{m['season']}W{m['week']}: Predicted={m['predicted']}, Actual={m['actual']}")
    if m['note']:
        print(f"    Note: {m['note']}")
