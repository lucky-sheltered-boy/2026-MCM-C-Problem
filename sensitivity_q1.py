"""
Sensitivity Analysis for Q1 Model (MCMC Prior Weights)

This script analyzes how sensitive the fan vote estimates are to the choice of 
heuristic weights used in the Week 1 Prior.

Parameters to test:
1. w_bottom1: Weight for "Times Saved from Bottom 1"
2. w_partner: Weight for "Partner Rank"

Target Season: Season 27 (Bobby Bones season - highly sensitive to fan votes)
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from data_loader import DWTSDataLoader
    from mcmc_sampler import MCMCSampler
except ImportError:
    # Fallback if running from a different cwd
    sys.path.insert(0, 'src')
    from data_loader import DWTSDataLoader
    from mcmc_sampler import MCMCSampler

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
if not os.path.exists('figures'):
    os.makedirs('figures')


def get_weighted_prior(df, season, survivor_names, weights):
    """
    Modified prior calculator with adjustable weights
    weights = {'b1': 2.0, 'b2': 1.0, 'partner': 3.0}
    """
    season_df = df[df['season'] == season]
    scores = []
    
    w_b1 = weights.get('b1', 2.0)
    w_b2 = weights.get('b2', 1.0)
    w_partner = weights.get('partner', 3.0)
    
    for name in survivor_names:
        row = season_df[season_df['celebrity_name'] == name]
        if row.empty:
            scores.append(1.0)
            continue
        row = row.iloc[0]
        
        b1 = row.get('total_fan_saves_bottom1', 0)
        b2 = row.get('total_fan_saves_bottom2', 0)
        partner_rank = row.get('partner_avg_placement', 0.5)
        
        # Formula
        score = 1.0 + (b1 * w_b1) + (b2 * w_b2) + (w_partner * (1.0 - partner_rank))
        scores.append(score)
        
    scores = np.array(scores)
    return scores / scores.sum()

def calculate_smoothness_distance(prev_votes, curr_votes, prev_names, curr_names):
    common_names = set(prev_names) & set(curr_names)
    if len(common_names) == 0: return float('inf')
    
    prev_common = []
    curr_common = []
    for name in common_names:
        prev_idx = prev_names.index(name)
        curr_idx = curr_names.index(name)
        prev_common.append(prev_votes[prev_idx])
        curr_common.append(curr_votes[curr_idx])
        
    p = np.array(prev_common)
    c = np.array(curr_common)
    return np.sqrt(np.sum(((p/(p.sum()+1e-9)) - (c/(c.sum()+1e-9))) ** 2))

def run_simulation(season, weights, loader, sampler):
    """
    Run the sequential smooth process for one season with specific weights
    """
    np.random.seed(season * 123) # Fixed seed for reproducibility
    
    voting_method = loader.get_voting_method(season)
    processed = loader.process_season(season)
    max_week = processed['max_week']
    
    prev_votes = None
    prev_names = None
    
    results = {}
    
    for week in range(1, max_week + 1):
        if week not in processed['weeks']: continue
        week_data = processed['weeks'][week]
        if week_data.get('is_finale', False): continue
        
        survivor_names = week_data['survivor_names']
        judge_share = week_data['judge_share']
        elim_indices = week_data.get('eliminated_indices_in_survivors', [])
        
        # 1. Determine Prior / Projection
        if prev_votes is None:
            # Week 1: Use Weighted Prior
            current_base = get_weighted_prior(loader.raw_data, season, survivor_names, weights)
        else:
            # Projection
            prev_map = {n: v for n, v in zip(prev_names, prev_votes)}
            temp = []
            for n in survivor_names:
                temp.append(prev_map.get(n, 1.0/len(survivor_names)))
            temp = np.array(temp)
            current_base = temp / temp.sum()
            
        # 2. No elimination?
        if not elim_indices:
            selected_votes = current_base
        else:
            # 3. MCMC Sampling
            # For sensitivity analysis, we can cheat a bit for speed:
            # Use MCMC but filter quickly
            elim_idx = elim_indices[0] 
            
            # Use sampler
            # We assume sampler is stateless
            mcmc_samples = sampler.sample_week(judge_share, elim_idx, voting_method, 
                                             week_data.get('judge_ranks', None))
            
            # Simple Consistency Check
            valid_samples = []
            if mcmc_samples is not None:
                # We need to reimplement a simple check here or assume sampler returns valid?
                # sampler.sample_week returns ALL samples, need to filter
                # For speed in this script, let's pick the sample closest to 'current_base' 
                # that satisfies consistency.
                
                # Filter locally
                for s in mcmc_samples:
                    # Quick consistency check (copy-paste logic simplified)
                    is_consistent = False
                    if voting_method == 'percentage':
                        combined = 0.5 * judge_share + 0.5 * s
                        if np.argmin(combined) == elim_idx: is_consistent = True
                    elif voting_method == 'rank':
                        # Simplified rank check
                        j_rank = len(s) - np.argsort(np.argsort(judge_share))
                        f_rank = len(s) - np.argsort(np.argsort(s))
                        if np.argmax(j_rank + f_rank) == elim_idx: is_consistent = True
                        
                    if is_consistent:
                        valid_samples.append(s)
            
            if not valid_samples:
                # Fallback
                selected_votes = current_base
            else:
                # Select smoothest (closest to current_base/prev_week)
                dists = [np.sqrt(np.sum((s - current_base)**2)) for s in valid_samples]
                selected_votes = valid_samples[np.argmin(dists)]
                
        # Update state
        prev_votes = selected_votes
        prev_names = survivor_names
        
        # Record Bobby Bones (for S27) or Winner
        results[week] = {n: v for n, v in zip(survivor_names, selected_votes)}
        
    return results

def sensitivity_analysis_prior():
    print("Running Sensitivity Analysis on Prior Weights (Season 27)...")
    
    loader = DWTSDataLoader('engineered_data.csv')
    loader.load_data() # Explicitly load data
    sampler = MCMCSampler(n_iterations=500, burn_in=100) # Lower samples for speed
    
    season_id = 27
    target_contestant = "Bobby Bones"
    
    # 1. Vary w_partner (Partner Rank Weight)
    # 0 = No partner effect, 10 = Strong partner effect
    partner_weights = [0, 1, 3, 5, 8]
    bobby_votes_partner = []
    
    print("\nTest 1: Varying Partner Weight (w_partner)")
    for w in partner_weights:
        weights = {'b1': 2.0, 'b2': 1.0, 'partner': float(w)}
        res = run_simulation(season_id, weights, loader, sampler)
        
        # Get Average Fan Vote for Bobby across all weeks
        bobby_series = [res[w][target_contestant] for w in res if target_contestant in res[w]]
        avg_vote = np.mean(bobby_series) if bobby_series else 0
        bobby_votes_partner.append(avg_vote)
        print(f"  w_partner={w}, {target_contestant} Avg Vote={avg_vote:.4f}")
        
    # 2. Vary w_bottom (Historical Bottom Save Weight)
    # 0 = Ignore history, 5 = Heavily use history
    bottom_weights = [0, 1, 2, 4, 6]
    bobby_votes_bottom = []
    
    print("\nTest 2: Varying Bottom Save Weight (w_bottom1)")
    for w in bottom_weights:
        weights = {'b1': float(w), 'b2': float(w)/2, 'partner': 3.0}
        res = run_simulation(season_id, weights, loader, sampler)
        
        bobby_series = [res[w][target_contestant] for w in res if target_contestant in res[w]]
        avg_vote = np.mean(bobby_series) if bobby_series else 0
        bobby_votes_bottom.append(avg_vote)
        print(f"  w_bottom1={w}, {target_contestant} Avg Vote={avg_vote:.4f}")
        
    # Plotting - Beautified
    
    # Set style context
    sns.set_context("talk")
    sns.set_style("white", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Modern Color Palette
    color_main = '#2c3e50' # Dark Blue/Grey
    color_accent = '#e74c3c' # Alizarin Red
    color_line1 = '#3498db' # Peter river
    color_line2 = '#16a085' # Green Sea
    
    # Plot 1: Partner Weight
    ax1.plot(partner_weights, bobby_votes_partner, marker='o', markersize=10, 
             linestyle='-', linewidth=3, color=color_line1)
    
    # Highlight Baseline
    ax1.axvline(3.0, color=color_accent, linestyle='--', linewidth=2)
    ax1.text(3.1, np.mean(bobby_votes_partner), 'Baseline (3.0)', color=color_accent, 
             fontsize=12, fontweight='bold', rotation=90, va='center')
    
    # Aesthetics 1
    ax1.set_title('Robustness: Partner Prior Weight', fontsize=18, pad=15)
    ax1.set_xlabel(r'Partner Weight ($w_{partner}$)', fontsize=14)
    ax1.set_ylabel('Est. Fan Share (Bobby Bones)', fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Annotate values
    for x, y in zip(partner_weights, bobby_votes_partner):
        ax1.annotate(f'{y:.1%}', (x, y), xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontsize=11, fontweight='bold', color=color_main)
    
    # Plot 2: Bottom Save Weight
    ax2.plot(bottom_weights, bobby_votes_bottom, marker='s', markersize=10, 
             linestyle='-', linewidth=3, color=color_line2)
    
    # Highlight Baseline
    ax2.axvline(2.0, color=color_accent, linestyle='--', linewidth=2)
    ax2.text(2.1, np.mean(bobby_votes_bottom), 'Baseline (2.0)', color=color_accent, 
             fontsize=12, fontweight='bold', rotation=90, va='center')
    
    # Aesthetics 2
    ax2.set_title('Robustness: Historical Save Prior', fontsize=18, pad=15)
    ax2.set_xlabel(r'Bottom Save Weight ($w_{bottom1}$)', fontsize=14)
    ax2.set_ylabel('Est. Fan Share (Bobby Bones)', fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Annotate values
    for x, y in zip(bottom_weights, bobby_votes_bottom):
        ax2.annotate(f'{y:.1%}', (x, y), xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontsize=11, fontweight='bold', color=color_main)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('figures/Task1_Sensitivity_Model.png', dpi=300, bbox_inches='tight')
    print("Saved beautified plot to figures/Task1_Sensitivity_Model.png")

    
if __name__ == '__main__':
    sensitivity_analysis_prior()
