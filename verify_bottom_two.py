"""验证不一致周是否因为评委选择机制"""
import pickle
import numpy as np

with open('results/estimation_results.pkl', 'rb') as f:
    r = pickle.load(f)

print("验证：第28季起所有不一致周的实际淘汰者是否在垫底两人中？")
print("=" * 60)

# 检查排名法赛季的不一致周
rank_seasons = list(range(1, 3)) + list(range(28, 35))
all_ok = True
mismatch_count = 0

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
            mismatch_count += 1
            # 检查实际淘汰者是否在垫底两人中
            bottom_two_idx = np.argsort(-total_ranks)[:2]
            bottom_two = [survivor_names[i] for i in bottom_two_idx]
            in_bottom = actual in bottom_two
            
            print(f"S{season}W{week}:")
            print(f"  垫底两人: {bottom_two}")
            print(f"  预测淘汰: {predicted}")
            print(f"  实际淘汰: {actual}")
            print(f"  在垫底两人中? {'✓ Yes' if in_bottom else '✗ No'}")
            print()
            
            if not in_bottom:
                all_ok = False

print("=" * 60)
print(f"总不一致周数: {mismatch_count}")
print(f"所有实际淘汰者都在垫底两人中? {'✓ Yes - 评委选择机制有效!' if all_ok else '✗ No'}")
