"""
Sensitivity Analysis for Q3 (Factor Analysis via Ridge Regression)

Objective:
Verify the stability of the "Factor Importance" findings.
We used Ridge Regression with alpha=1.0. We need to ensure that the 
conclusions (e.g., "Age negatively affects Judge Score", "Partner positively affects Judge Score")
are not artifacts of the specific regularization strength or feature scaling.

Methodology:
1. Vary Ridge Regularization parameter (alpha) from 0.01 (near OLS) to 100.0 (strong regularization).
2. Measure stability of:
   - Model Predictive Power (R^2)
   - Feature Coefficient Ranks (Do the top factors stay on top?)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Import Q3 module
try:
    import analysis_q3_factors as q3
except ImportError:
    print("Could not import analysis_q3_factors.py")
    sys.exit(1)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
if not os.path.exists('figures'):
    os.makedirs('figures')


def run_ridge_sensitivity(X, y, feature_names, target_name):
    """
    Run Ridge regression with varying alphas and track metrics.
    """
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    r2_scores = []
    r2_stds = []
    coef_history = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nAnalyzing Sensitivity for Target: {target_name}")
    print(f"{'Alpha':^10} | {'R2 (CV)':^10} | {'Top Feature':^30}")
    print("-" * 60)
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        
        # 1. Evaluate Performance (CV)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        r2_scores.append(scores.mean())
        r2_stds.append(scores.std())
        
        # 2. Evaluate Coefficients (Full Fit)
        model.fit(X_scaled, y)
        coefs = model.coef_
        coef_history.append(coefs)
        
        # Identify top feature
        top_idx = np.argmax(np.abs(coefs))
        top_feat = feature_names[top_idx]
        
        print(f"{alpha:^10.2f} | {scores.mean():^10.4f} | {top_feat:<30}")
        
    return alphas, r2_scores, np.array(coef_history)

def plot_sensitivity(alphas, r2_j, coefs_j, r2_f, coefs_f, feature_names):
    """
    Plot R2 stability and Coefficient Paths - Beautified
    """
    sns.set_context("talk")
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Plot 1: Predictive Power Stability ---
    # Colors
    c_judge = '#8e44ad' # Purple
    c_fan = '#e67e22'   # Orange
    
    ax1.semilogx(alphas, r2_j, marker='o', markersize=9, linewidth=3, color=c_judge, 
                 label='Judge Score Model ($R^2$)')
    ax1.semilogx(alphas, r2_f, marker='x', markersize=9, linewidth=3, color=c_fan, 
                 label='Fan Vote Model ($R^2$)', linestyle=':')
    
    # Annotate Values for Judge Model
    for i, (x, y) in enumerate(zip(alphas, r2_j)):
        if i % 2 == 0 or i == len(alphas)-1: # Skip some to avoid clutter
            ax1.annotate(f"{y:.2f}", (x, y), xytext=(0, 10), textcoords='offset points', 
                         ha='center', fontsize=11, color=c_judge, fontweight='bold')

    ax1.set_xlabel('Regularization Strength (Alpha) - Log Scale', fontsize=14)
    ax1.set_ylabel('Model Accuracy ($CV-R^2$)', fontsize=14)
    ax1.set_title('Model Performance Stability', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(loc='center right', frameon=True, shadow=True)
    ax1.set_ylim(-0.1, 0.4) # Show contrast clearly
    sns.despine(ax=ax1)

    # --- Plot 2: Feature Coefficient Stability (Judge Model) ---
    # Identify top 5 factors by absolute coefficient at Alpha=1.0 (index 2)
    ref_idx = 2 if len(alphas) > 2 else 0
    ref_coefs = coefs_j[ref_idx] 
    top_indices = np.argsort(np.abs(ref_coefs))[-5:] # Top 5
    
    # Color palette
    palette = sns.color_palette("viridis", n_colors=5)
    
    for i, idx in enumerate(reversed(top_indices)): # Plot most important last (on top)
        feat_name = feature_names[idx]
        feat_values = coefs_j[:, idx]
        # Clean naming
        display_name = feat_name.replace('celebrity_', '').replace('industry_', 'Ind: ').title()
        
        ax2.semilogx(alphas, feat_values, marker='o', markersize=6, linewidth=2.5, 
                     color=palette[i], label=display_name)
        
        # Label the lines directly at the end
        ax2.text(alphas[-1]*1.1, feat_values[-1], display_name, va='center', fontsize=11, 
                 color=palette[i], fontweight='bold')
        
    ax2.set_xlabel('Regularization Strength (Alpha) - Log Scale', fontsize=14)
    ax2.set_ylabel('Standardized Coefficient', fontsize=14)
    ax2.set_title('Stability of Top Factors (Judge Model)', fontsize=18, fontweight='bold', pad=15)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    sns.despine(ax=ax2)
    
    plt.tight_layout(pad=3.0)
    save_path = 'figures/Task3_Sensitivity_Factors.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved beautified plot to {save_path}")

def main():
    # 1. Reuse data loading logic from Q3
    # Use internal functions to avoid running main()'s print statements if possible
    # But Q3 `main` runs automatically on import if not guarded?
    # Checked file content: `if __name__ == "__main__":` guard exists. Good.
    
    print("Loading and preparing data using Q3 logic...")
    df, _ = q3.load_and_prepare_data()
    df = q3.calculate_celebrity_fan_votes(df)
    
    # 2. Prepare Feature Matrices
    X_j, y_j, _, feats_j = q3.prepare_features(df, 'avg_judge_score')
    # Note: prepare_features returns numeric features list, but X includes dummies
    # We need full feature names
    feature_names = X_j.columns.tolist()
    
    X_f, y_f, _, _ = q3.prepare_features(df, 'avg_fan_vote')
    
    # 3. Run Analysis
    # Judge Model
    alphas, r2_j, coefs_j = run_ridge_sensitivity(X_j, y_j, feature_names, "Judge Scores")
    
    # Fan Model
    _, r2_f, coefs_f = run_ridge_sensitivity(X_f, y_f, feature_names, "Fan Votes")
    
    # 4. Plot
    plot_sensitivity(alphas, r2_j, coefs_j, r2_f, coefs_f, feature_names)

if __name__ == '__main__':
    main()
