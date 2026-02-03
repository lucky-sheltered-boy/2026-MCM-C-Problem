"""
Sensitivity Analysis: Robustness Check of the Proposed System

This script performs sensitivity analysis on two key parameters of the DWTS 2.0 System:
1. Judge Weight (alpha): varying the balance between Judge Score and Fan Vote.
   - Base value: 0.5 (50/50 split)
   - Range: 0.3 to 0.7
   - Metric: Impact on elimination results (Consistency with original history).

2. Golden Save Threshold (beta): varying the "Judge Rank" requirement to trigger a save.
   - Base value: 0.5 (Top 50% of Judge Rank)
   - Range: 0.2 to 0.8
   - Metric: Number of total Golden Saves triggered across all seasons.

"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# Create figures directory if not exists
if not os.path.exists('figures'):
    os.makedirs('figures')

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    print("Loading Data...")
    mcmc_df = pd.read_csv('results/mcmc_smooth_results.csv')
    raw_df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
    return mcmc_df, raw_df

def simulate_weighted_system(mcmc_df, raw_df, judge_weight=0.5):
    """
    Simulates the elimination process with a variable weight for judge scores.
    Score = weight * Judge% + (1-weight) * Fan%
    """
    results = []
    
    # Cache raw data by season to speed up lookups
    raw_by_season = {s: raw_df[raw_df['season'] == s] for s in raw_df['season'].unique()}
    
    for _, row in mcmc_df.iterrows():
        try:
            names = ast.literal_eval(row['survivor_names'])
            votes = ast.literal_eval(row['fan_votes_mean'])
            eliminated = row['eliminated']
        except:
            continue
            
        if pd.isna(eliminated) or 'None' in str(eliminated):
            continue
            
        season = row['season']
        week = row['week']
        season_raw = raw_by_season.get(season)
        
        if season_raw is None: continue

        # Get Judge Scores
        judge_scores = {}
        for name in names:
            celeb_row = season_raw[season_raw['celebrity_name'] == name]
            if not celeb_row.empty:
                score = 0
                count = 0
                for j in range(1, 5):
                    col = f'week{week}_judge{j}_score'
                    if col in celeb_row.columns:
                        val = celeb_row[col].values[0]
                        if pd.notna(val) and val > 0:
                            score += val
                            count += 1
                if score > 0:
                    judge_scores[name] = score

        if not judge_scores: continue
        
        # Normalize to percentages
        total_judge = sum(judge_scores.values())
        total_fan = sum(votes)
        
        if total_judge == 0 or total_fan == 0: continue
        
        combined_scores = {}
        for i, name in enumerate(names):
            if i < len(votes) and name in judge_scores:
                # Weighted Sum
                judge_pct = (judge_scores[name] / total_judge) * 100
                fan_pct = (votes[i] / total_fan) * 100
                
                # SENSITIVITY PARAMETER APPLIED HERE
                combined_scores[name] = judge_weight * judge_pct + (1 - judge_weight) * fan_pct
        
        if combined_scores:
            lowest_scorer = min(combined_scores, key=combined_scores.get)
            
            # Check if the result changed from actual history
            is_different = (lowest_scorer != eliminated)
            
            results.append({
                'season': season,
                'week': week,
                'eliminated_actual': eliminated,
                'lowest_scorer_simulated': lowest_scorer,
                'is_different': is_different,
                'judge_scores': judge_scores, # store for next step
                'n_contestants': len(names)
            })
            
    return results

def sensitivity_judge_weight(mcmc_df, raw_df):
    """
    Vary Judge Weight from 0.0 to 1.0
    Measure: Percentage of eliminations that differ from historical result
    (Assuming historical result is the "ground truth" of what happened, 
     divergence implies the model is changing the outcome)
    """
    print("\n[Analysis 1] Sensitivity to Judge Weight (alpha)")
    
    weights = np.linspace(0.0, 1.0, 21) # 0.0, 0.05, ..., 1.0
    change_rates = []
    
    for w in weights:
        res = simulate_weighted_system(mcmc_df, raw_df, judge_weight=w)
        if not res:
            change_rates.append(0)
            continue
        
        n_diff = sum(1 for r in res if r['is_different'])
        rate = n_diff / len(res) * 100
        change_rates.append(rate)
        # print(f"  Weight={w:.2f}, Change Rate={rate:.2f}%")
        
    # Plotting - Beautified
    sns.set_context("talk")
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Line
    ax.plot(weights, change_rates, marker='o', markersize=8, linewidth=3, 
            color='#2c3e50', label='Outcome Divergence')
            
    # Highlight Proposed Point
    idx_05 = int(np.abs(weights - 0.5).argmin()) # Fix index to int
    ax.scatter([0.5], [change_rates[idx_05]], s=200, color='#e74c3c', zorder=5, 
               edgecolors='white', linewidth=2, label=r'Proposed ($\alpha$=0.5)')
    
    # Text Annotation
    ax.annotate(f"Split Point\nChange Rate: {change_rates[idx_05]:.1f}%", 
                (0.5, change_rates[idx_05]), xytext=(40, -40), textcoords='offset points', 
                arrowprops=dict(arrowstyle="->", color='#e74c3c'),
                fontsize=11, color='#c0392b', fontweight='bold')
    
    # Aesthetics
    ax.set_title('System Impact: Judge Weight Sensitivity', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(r'Judge Weight Parameter ($\alpha$)', fontsize=14)
    ax.set_ylabel('Outcome Divergence from History (%)', fontsize=14)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('figures/Task4_Sensitivity_Judge_Weight.png', dpi=300)
    print("  Saved beautified plot to figures/Task4_Sensitivity_Judge_Weight.png")
    return weights, change_rates

def simulate_golden_save_counts(base_results, raw_df, threshold_ratio):
    """
    Simulates how many times Golden Save triggers given a threshold ratio.
    """
    total_saves = 0
    # Group by season to enforce "cards per season" limit logic if needed,
    # but for sensitivity we mainly care about "Demand" for saves.
    # To be strictly comparable to Q4, we should enforce the limit.
    
    results_by_season = defaultdict(list)
    for r in base_results:
        results_by_season[r['season']].append(r)
        
    for season, weeks in results_by_season.items():
        cards_remaining = 3 # Const
        
        for week_data in sorted(weeks, key=lambda x: x['week']):
            if cards_remaining <= 0: continue
            
            # The simulated 'eliminated' person is the one with lowest score in THIS simulation
            # But for Golden Save logic in Q4 we used 'eliminated_current'
            # Let's use the 'lowest_scorer_simulated' as the candidate for elimination
            candidate = week_data['lowest_scorer_simulated']
            judge_scores = week_data['judge_scores']
            n = week_data['n_contestants']
            
            if candidate not in judge_scores: continue
            
            # Calculate Judge Rank
            sorted_by_judge = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
            judge_rank_map = {name: i+1 for i, (name, _) in enumerate(sorted_by_judge)}
            
            cand_rank = judge_rank_map.get(candidate, n)
            
            # SENSITIVITY PARAMETER APPLIED HERE
            # If threshold is 0.5, rank must be <= 0.5 * n (Top 50%)
            cutoff_rank = n * threshold_ratio
            
            if cand_rank <= cutoff_rank:
                total_saves += 1
                cards_remaining -= 1
                
    return total_saves

def sensitivity_save_threshold(mcmc_df, raw_df):
    """
    Vary Golden Save Threshold from 0.1 (Top 10%) to 0.9 (Top 90%)
    Measure: Total Number of Golden Saves Triggered
    """
    print("\n[Analysis 2] Sensitivity to Golden Save Threshold (beta)")
    
    # First get the base simulation results at alpha=0.5
    base_res = simulate_weighted_system(mcmc_df, raw_df, judge_weight=0.5)
    
    ratios = np.linspace(0.1, 0.9, 17) # 0.1, 0.15, ..., 0.9
    save_counts = []
    
    for r in ratios:
        count = simulate_golden_save_counts(base_res, raw_df, threshold_ratio=r)
        save_counts.append(count)
        # print(f"  Threshold={r:.2f}, Total Saves={count}")
        
    # Plotting - Beautified
    sns.set_context("talk")
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Line
    ax.plot(ratios, save_counts, marker='s', markersize=8, linewidth=3, 
            color='#27ae60', label='Golden Save Frequency')
            
    # Highlight Proposed Point (0.5)
    idx_05 = int(np.abs(ratios - 0.5).argmin())
    val_05 = save_counts[idx_05]
        
    ax.scatter([0.5], [val_05], s=200, color='#f39c12', zorder=5, 
               edgecolors='white', linewidth=2, label=r'Proposed ($\beta$=50%)')
               
    ax.annotate(f"Optimal Threshold\n{val_05} Saves (Total)", 
                (0.5, val_05), xytext=(-20, 40), textcoords='offset points', 
                arrowprops=dict(arrowstyle="->", color='#f39c12'),
                fontsize=11, color='#d35400', fontweight='bold')

    # Aesthetics
    ax.set_title('System Impact: Golden Save Threshold', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Judge Rank Threshold (Top X%)', fontsize=14)
    ax.set_ylabel('Total Trigger Count (All Seasons)', fontsize=14)
    
    # Helper lines describing scarcity
    ax.axhline(val_05, color='#f39c12', linestyle=':', alpha=0.5)
    
    ax.legend(loc='upper left', frameon=True, shadow=True)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('figures/Task4_Sensitivity_Save_Threshold.png', dpi=300)
    print("  Saved beautified plot to figures/Task4_Sensitivity_Save_Threshold.png")
    return ratios, save_counts

def main():
    mcmc_df, raw_df = load_data()
    
    # 1. Judge Weight Sensitivity
    sensitivity_judge_weight(mcmc_df, raw_df)
    
    # 2. Golden Save Threshold Sensitivity
    sensitivity_save_threshold(mcmc_df, raw_df)
    
    print("\nSensitivity Analysis Complete.")

if __name__ == '__main__':
    main()
