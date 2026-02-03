"""
Sensitivity Analysis for Q2 (Counterfactual Analysis)

Objective:
Test the robustness of the "Method Disagreement" (Rank vs Percent) finding.
Since Fan Votes are estimates, we inject noise into them to see if the
discrepancy between Rank and Percent methods is structural or accidental.

Methodology:
1. Monte Carlo Simulation with Noise injection.
   New_Vote = Original_Vote * (1 + GaussianNoise(0, sigma))
2. Re-calculate eliminations for both methods.
3. Measure how stable the "Disagreement Rate" is as noise increases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import necessary functions from the original script
try:
    import analysis_q2_counterfactual as q2
except ImportError:
    print("Could not import analysis_q2_counterfactual.py")
    sys.exit(1)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
if not os.path.exists('figures'):
    os.makedirs('figures')

def add_noise_to_votes(votes_array, noise_level=0.0):
    """
    Applies Gaussian noise to vote shares and renormalizes.
    new_v = v * (1 + N(0, sigma))
    """
    if noise_level == 0:
        return votes_array
    
    noise = np.random.normal(0, noise_level, size=len(votes_array))
    # multiplicative noise ensures non-negativity (mostly)
    noisy_votes = votes_array * (1 + noise)
    # Clip to be safe and avoid negative votes
    noisy_votes = np.maximum(noisy_votes, 1e-6)
    # Renormalize
    return noisy_votes / noisy_votes.sum()

def run_sensitivity_simulation(mcmc_results, noise_levels):
    """
    Runs the Q2 logic multiple times with varying noise levels on fan votes.
    Attributes to track:
    1. Agreement Rate between Rank vs Percent (Structural Robustness)
    2. Rank Method's Match Rate with History (Historical Robustness)
    """
    
    agreement_stability = []
    rank_accuracy_stability = []
    
    print(f"Running Q2 Sensitivity Analysis over {len(noise_levels)} noise levels...")
    
    for sigma in noise_levels:
        # Run multiple trials per sigma to average out random noise
        n_trials = 20 if sigma > 0 else 1 
        trial_agreements = []
        trial_accuracies = []
        
        for _ in range(n_trials):
            total_weeks = 0
            agree_weeks = 0
            rank_matches = 0
            
            # Iterate through all seasons/weeks
            for season in mcmc_results:
                for week in mcmc_results[season]:
                    week_data = mcmc_results[season][week]
                    
                    if week_data.get('is_no_elimination', False) or week_data.get('is_finale', False):
                        continue
                        
                    # Get base data
                    survivor_names = week_data['survivor_names']
                    judge_share = week_data['judge_share']
                    base_fan_votes = week_data['fan_votes_mean']
                    
                    if len(survivor_names) < 2: continue
                    
                    # INJECT NOISE
                    noisy_fan_votes = add_noise_to_votes(base_fan_votes, sigma)
                    
                    # Re-calculate methods
                    # 1. Rank Method
                    rank_res = q2.calculate_rank_method(survivor_names, judge_share, noisy_fan_votes)
                    elim_rank = q2.get_eliminated_by_rank(rank_res, week_data['n_eliminations'])
                    
                    # 2. Percent Method
                    pct_res = q2.calculate_percent_method(survivor_names, judge_share, noisy_fan_votes)
                    elim_pct = q2.get_eliminated_by_percent(pct_res, week_data['n_eliminations'])
                    
                    # Metrics
                    # Note: elim_rank/pct are lists of names
                    if sorted(elim_rank) == sorted(elim_pct):
                        agree_weeks += 1
                        
                    # Check against history
                    actual = week_data['eliminations']
                    # For accuracy, we only check if the predicted set matches actual set exactly
                    if sorted(elim_rank) == sorted(actual):
                        rank_matches += 1
                        
                    total_weeks += 1
            
            if total_weeks > 0:
                trial_agreements.append(agree_weeks / total_weeks)
                trial_accuracies.append(rank_matches / total_weeks)
        
        # Average across trials
        avg_agree = np.mean(trial_agreements)
        avg_acc = np.mean(trial_accuracies)
        
        agreement_stability.append(avg_agree)
        rank_accuracy_stability.append(avg_acc)
        print(f"  Sigma={sigma:.2f}: Agreement={avg_agree:.4f}, Rank Accuracy (vs History)={avg_acc:.4f}")
        
    return agreement_stability, rank_accuracy_stability

def plot_results(noise_levels, agreement, accuracy):
    # Modern Style - User Requested Beautification
    sns.set_context("talk")
    sns.set_style("white", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors
    col_agree = '#2980b9'  # Strong Blue
    col_acc = '#27ae60'    # Strong Green
    
    # Plot Lines
    ax.plot(noise_levels, agreement, marker='o', markersize=9, 
            label='Method Agreement (Rank vs %)', linewidth=3, color=col_agree)
    ax.plot(noise_levels, accuracy, marker='D', markersize=8, 
            label='Rank Accuracy (vs History)', linewidth=2.5, linestyle='--', color=col_acc)
    
    # Highlight No-Noise Point
    ax.scatter([0], agreement[0], color='#c0392b', s=150, zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f"Baseline: {agreement[0]:.1%}", (0, agreement[0]), xytext=(15, 0), 
                textcoords='offset points', color=col_agree, fontweight='bold', va='center', fontsize=12)

    # Highlight High-Noise Point
    ax.scatter([noise_levels[-1]], agreement[-1], color='#c0392b', s=100, zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f"Noise 30%: {agreement[-1]:.1%}", (noise_levels[-1], agreement[-1]), xytext=(0, 15), 
                textcoords='offset points', ha='center', color=col_agree, fontweight='bold', fontsize=10)
    
    # Labels & Title
    ax.set_xlabel(r'Noise Injection Level ($\sigma$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Consistency Metric', fontsize=14, labelpad=10)
    ax.set_title('Robustness: Method Agreement vs Data Noise', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0.5, 0.75)  # Focus on the relevant range for better detail
    
    # Legend
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, shadow=True, fontsize=12)
    
    # Remove Spines
    sns.despine(trim=True)
    
    # Save
    save_path = 'figures/Task2_Sensitivity_Method_Comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved beautified plot to {save_path}")

def main():
    # Load data using Q2's loader
    # Note: load_mcmc_results logic is complex in Q2, let's just use it
    try:
        mcmc_results = q2.load_mcmc_results() 
    except FileNotFoundError:
        print("Data file not found. Ensure you are in the correct directory.")
        return

    # Define noise levels (0% to 30% perturbation)
    noise_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    agree, acc = run_sensitivity_simulation(mcmc_results, noise_vals)
    
    plot_results(noise_vals, agree, acc)

if __name__ == '__main__':
    main()
