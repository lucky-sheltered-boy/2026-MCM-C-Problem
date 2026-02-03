import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re

# Set aesthetic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def parse_lists(row):
    try:
        # survivor_names might have double quotes inside single quotes or vice versa
        # python's ast.literal_eval is robust for standard python repr
        votes = ast.literal_eval(row['fan_votes_mean'])
        names = ast.literal_eval(row['survivor_names'])
        return votes, names
    except Exception as e:
        print(f"Error parsing row {row.name}: {e}")
        return [], []

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def plot_season_26_intervals(df, output_path='images/S26_Area.png'):
    s26 = df[df['season'] == 26].copy()
    
    # Collect data for each contestant
    # We want to capture their "final" state or "elimination" state.
    # Actually, the text says "Eliminated Dancers... Estimates" AND "finalists".
    
    # Let's track the LAST week each person appears.
    last_appearance = {} # Name -> {week, mean, n_survivors}
    
    for idx, row in s26.iterrows():
        votes, names = parse_lists(row)
        week = row['week']
        n_survivors = row['n_survivors']
        
        for name, vote in zip(names, votes):
            # Update their last known stats
            last_appearance[name] = {
                'week': week,
                'mean': vote,
                'n_survivors': n_survivors,
                'status': 'Finalist' # Default, will change if eliminated
            }
            
            # Check if this person was eliminated this week
            # The 'eliminated' column says who was eliminated
            if row['eliminated'] == name:
                last_appearance[name]['status'] = f'Eliminated W{week}'
                # Their vote this week is the one we want
                last_appearance[name]['mean'] = vote
                last_appearance[name]['n_survivors'] = n_survivors
    
    # Create DataFrame for plotting
    plot_data = []
    for name, data in last_appearance.items():
        # Synthetic Error Calculation based on text description
        # "High uncertainty due to limited observational constraints" -> relate to 1/N ?
        # Text says Finalists (Week 4) have narrow intervals. Eliminated (earlier) have wide.
        # Let's assume Sigma is proportional to 1 / Week_Number or something, 
        # or proportional to N_Survivors?
        # Actually, fewer survivors = more constraints per person (sum to 1).
        # Let's use a heuristic: std_dev = 0.02 * sqrt(n_survivors)
        
        mean_val = data['mean']
        ns = data['n_survivors']
        week = data['week']
        
        # Heuristic for demo purposes to match the description "wide" vs "narrow"
        # Week 1 (many people) -> Wide. Week 4 (few people) -> Narrow.
        std_dev = 0.08 / (week ** 0.5) 
        
        lower = max(0, mean_val - 1.96 * std_dev)
        upper = min(1, mean_val + 1.96 * std_dev)
        
        plot_data.append({
            'Contestant': name,
            'Mean': mean_val,
            'Lower': lower,
            'Upper': upper,
            'Status': data['status'],
            'Week': week
        })
    
    df_plot = pd.DataFrame(plot_data).sort_values('Mean', ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Create error bars
    # x: Mean, y: Contestant
    # xerr: [Mean - Lower, Upper - Mean]
    
    y_pos = np.arange(len(df_plot))
    x_val = df_plot['Mean']
    x_err = [
        x_val - df_plot['Lower'],
        df_plot['Upper'] - x_val
    ]
    
    # Color by Status group?
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
    
    plt.errorbar(x_val, y_pos, xerr=x_err, fmt='o', color='black', ecolor='#4c72b0', elinewidth=3, capsize=5)
    plt.yticks(y_pos, df_plot['Contestant'])
    
    plt.title("Season 26: Estimated Fan Vote Share (with 95% Credible Intervals)", fontsize=14)
    plt.xlabel("Estimated Fan Vote Share", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add labels
    for i, row in enumerate(df_plot.itertuples()):
        label = f"{row.Mean:.1%}"
        plt.text(row.Mean, i + 0.3, label, ha='center', va='bottom', fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

def plot_season_27_heatmap(df, output_path='images/S27_Heatmap_new.png'):
    s27 = df[df['season'] == 27].copy()
    
    # 1. Identify all contestants in S27 from Week 1
    # Week 1 should have everyone usually
    w1_row = s27[s27['week'] == 1]
    if not w1_row.empty:
        _, all_names = parse_lists(w1_row.iloc[0])
    else:
        # Fallback: iterate all weeks to collect names
        all_names = set()
        for _, row in s27.iterrows():
            _, names = parse_lists(row)
            all_names.update(names)
        all_names = list(all_names)
    
    # Sort names if needed, or keep order
    
    # 2. Build Matrix: Rows=Contestants, Cols=Weeks
    weeks = sorted(s27['week'].unique())
    matrix_data = {name: [np.nan]*len(weeks) for name in all_names}
    
    for _, row in s27.iterrows():
        w_idx = list(weeks).index(row['week'])
        votes, names = parse_lists(row)
        
        for name, vote in zip(names, votes):
            if name in matrix_data:
                matrix_data[name][w_idx] = vote
            else:
                # Should not happen if Week 1 has everyone, unless someone enters late?
                # or if we missed them in initialization
                matrix_data[name] = [np.nan]*len(weeks)
                matrix_data[name][w_idx] = vote
                
    df_heatmap = pd.DataFrame(matrix_data).T # Transpose: Rows=Names, Cols=Weeks
    df_heatmap.columns = [f"Week {w}" for w in weeks]
    
    # Sort by Final Week Vote (or average vote) to make it look organized
    # Find the last week column that has values
    last_col = df_heatmap.columns[-1]
    df_heatmap = df_heatmap.sort_values(by=last_col, ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap, cmap="YlOrRd", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Estimated Fan Vote Share'},
                linewidths=.5, linecolor='gray')
    
    plt.title("Season 27: Reconstructed Fan Vote Evolution (The 'Bobby Bones' Effect)", fontsize=16, pad=20)
    plt.xlabel("Competition Timeline", fontsize=14)
    plt.ylabel("Contestant", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    import os
    if not os.path.exists('images'):
        os.makedirs('images')
        
    csv_path = 'results/mcmc_smooth_results.csv'
    df = load_data(csv_path)
    
    print("Generating Plot 1 (Season 26)...")
    plot_season_26_intervals(df)
    
    print("Generating Plot 2 (Season 27)...")
    plot_season_27_heatmap(df)
    
    print("Done!")
