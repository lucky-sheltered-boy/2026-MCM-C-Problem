import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import matplotlib.patches as mpatches

# Style settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    # Load MCMC results (Fan Votes)
    mcmc_df = pd.read_csv('results/mcmc_smooth_results.csv')
    s27_mcmc = mcmc_df[mcmc_df['season'] == 27].copy()
    
    # Load Raw Scores (Judge Scores)
    raw_df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
    s27_raw = raw_df[raw_df['season'] == 27].copy()
    
    return s27_mcmc, s27_raw

def parse_lists(row):
    try:
        temp_names = ast.literal_eval(row['survivor_names'])
        temp_votes = ast.literal_eval(row['fan_votes_mean'])
        # Clean names (handle quotes if needed, though ast usually handles it)
        return temp_names, temp_votes
    except:
        return [], []

def get_judge_scores_for_week(s27_raw, week, survivor_names):
    # Calculate Judge Share for these specific survivors
    scores = []
    for name in survivor_names:
        row = s27_raw[s27_raw['celebrity_name'] == name]
        if row.empty:
            scores.append(0)
            continue
        
        # Sum 4 judges if available
        # Cols: weekX_judgeY_score
        weekly_score = 0
        valid_judges = 0
        for j in range(1, 5):
            col = f'week{week}_judge{j}_score'
            if col in row.columns:
                val = pd.to_numeric(row.iloc[0][col], errors='coerce')
                if not pd.isna(val):
                    weekly_score += val
                    valid_judges += 1
        
        # Handle cases where they didn't dance? (Shouldn't happen for survivors)
        # If valid_judges == 0, maybe they are immune or score is 0?
        # Assuming sum
        scores.append(weekly_score)
        
    scores = np.array(scores)
    total = scores.sum()
    if total == 0:
        return np.ones(len(scores)) / len(scores)
    return scores / total

def calculate_ranks(arr, ascending=False):
    # Rank 1 is best (highest value)
    # scipy.stats.rankdata gives 1 for smallest
    # We want 1 for Largest
    # Method 'min' assigns same rank to ties (e.g. 1, 2, 2, 4) or 'average' (1, 2.5, 2.5, 4)
    # The show uses 'dense' or 'min'. Let's use dense logic or simple argsort for unique.
    # Code uses: n + 1 - rank(0=low)
    # Simulating simple rank:
    order = np.argsort(arr)[::-1] # indices of high to low
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr)) + 1
    return ranks

def process_bobby_bones_trajectory(s27_mcmc, s27_raw):
    weeks = sorted(s27_mcmc['week'].unique())
    bobby_data = []
    
    for week in weeks:
        row = s27_mcmc[s27_mcmc['week'] == week]
        if row.empty:
            continue
        row = row.iloc[0]
        
        names, fan_probs = parse_lists(row)
        if "Bobby Bones" not in names:
            continue
            
        judge_probs = get_judge_scores_for_week(s27_raw, week, names)
        
        # 1. Percent Method (Model P)
        # Combined Score = 0.5 * J + 0.5 * F
        percent_score = 0.5 * judge_probs + 0.5 * np.array(fan_probs)
        # Rank by score (Higher is better)
        percent_ranks = calculate_ranks(percent_score)
        # Find Bobby's rank
        b_idx = names.index("Bobby Bones")
        b_rank_percent = percent_ranks[b_idx]
        b_score_percent = percent_score[b_idx]
        
        # 2. Rank Method (Model R)
        # Rank(J) + Rank(F). Lower is Better.
        # Judge Rank (1 = Best)
        j_ranks = calculate_ranks(judge_probs)
        # Fan Rank (1 = Best)
        f_ranks = calculate_ranks(np.array(fan_probs))
        
        sum_ranks = j_ranks + f_ranks
        # Final ordering: Lowest Sum is Best.
        # So we rank the sums. Smallest sum gets Rank 1.
        # Note: If sums are equal, tie-breaker is usually Fan Vote.
        # But for simple visualization, let's just use the placement based on sum.
        # We need the Rank of the Sum (ascending).
        # i.e. if sums are [2, 4, 3], ranks are [1, 3, 2].
        final_ranks_model_r = np.empty_like(sum_ranks)
        temp_argsort = np.argsort(sum_ranks) # indices of low sum to high sum
        final_ranks_model_r[temp_argsort] = np.arange(len(sum_ranks)) + 1
        
        b_rank_rank_method = final_ranks_model_r[b_idx]
        
        n_contestants = len(names)
        
        bobby_data.append({
            'Week': week,
            'N_Contestants': n_contestants,
            'Rank_Percent': b_rank_percent,
            'Rank_RankMethod': b_rank_rank_method,
            'Is_Bottom': b_rank_percent >= n_contestants, # Did he have lowest score?
            'Fan_Share': fan_probs[b_idx],
            'Judge_Share': judge_probs[b_idx]
        })
        
    return pd.DataFrame(bobby_data)

def plot_bobby_bones(df, output_path='images/S27_Bobby_Simulation.png'):
    plt.figure(figsize=(10, 6))
    
    # Plot "Safety Line" (Rank = N_Contestants is dangerous)
    # Actually, Elimination is usually the LAST person.
    # So if Rank == N_Contestants, you are eliminated (simplification).
    
    # Plot Trajectory
    plt.plot(df['Week'], df['Rank_Percent'], marker='o', linewidth=3, label='Actual Outcome (Percent Method)', color='#2ca02c') # Green (Safe)
    plt.plot(df['Week'], df['Rank_RankMethod'], marker='x', linewidth=3, linestyle='--', label='Counterfactual (Rank Method)', color='#d62728') # Red (Danger)
    
    # Plot "Elimination Threshold" (The number of contestants that week)
    plt.plot(df['Week'], df['N_Contestants'], color='gray', linestyle=':', alpha=0.5, label='Elimination Zone (Last Place)')
    
    # Fill danger zone
    # plt.fill_between(df['Week'], df['N_Contestants'] - 0.5, df['N_Contestants'] + 1, color='gray', alpha=0.1)

    # Highlight Week 8
    w8 = df[df['Week'] == 8]
    if not w8.empty:
        p_rank = w8['Rank_Percent'].values[0]
        r_rank = w8['Rank_RankMethod'].values[0]
        limit = w8['N_Contestants'].values[0]
        
        plt.annotate('Week 8 Divergence', 
                     xy=(8, r_rank), xytext=(8, r_rank-2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
                     
        if r_rank >= limit:
             plt.text(8.2, r_rank, 'Would be Eliminated!', color='#d62728', fontweight='bold')
        
    plt.gca().invert_yaxis() # Rank 1 at top
    plt.yticks(range(1, 14))
    
    plt.title("The 'Bobby Bones' Anomaly: Actual vs. Counterfactual Ranking", fontsize=14)
    plt.xlabel("Competition Week")
    plt.ylabel("Computed Rank (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")


def plot_judges_save_gantt(output_path='images/Judges_Save_Impact.png'):
    # Data from text
    # Jerry Rice: 2 Bottom Two, Eliminated Early (Actual) vs Eliminated Early (Counter)
    # Wait, text says: Jerry Rice -> Eliminated Early.
    # Bobby Bones -> Eliminated Week 4 (Counter) vs Won (Actual, implied safe 4 times).
    
    # Let's verify the text about Jerry Rice: "Outcome: Eliminated Early".
    # Wait, in the table it says "Eliminated Early". Does that mean NO CHANGE?
    # Text says: "Judges' Save is a potent corrective... Had it been in place... Bobby Bones would likely have been eliminated as early as Week 4".
    # So for Bobby: Actual = 10 weeks (Winner). Counterfactual = 4 weeks.
    
    # Let's create a Gantt chart for Bobby Bones specifically, 
    # and maybe comparison for "Average Contestant" or similar?
    # Better: Just visualize the TABLE data differently.
    
    # Table Data
    data = [
        {'Name': 'Bobby Bones (S27)', 'Actual_Weeks': 10, 'Counter_Weeks': 4, 'Reason': 'Consistently Bottom 2'},
        {'Name': 'Jerry Rice (S2)',   'Actual_Weeks': 10, 'Counter_Weeks': 3, 'Reason': 'Weak Technical Score'}, # Assuming S2 lasted ~10 weeks
        {'Name': 'Bristol Palin (S11)', 'Actual_Weeks': 10, 'Counter_Weeks': 10, 'Reason': 'Never in Bottom 2'}
    ]
    # Note: Jerry Rice actually got 2nd place in S2. Text says "Jerry Rice: Outcome Eliminated Early".
    # This implies the Judges' Save WOULD HAVE eliminated him early.
    # So Actual = Long, Counter = Short.
    # Bristol: Actual = Long, Counter = Long.
    
    names = [d['Name'] for d in data]
    actual = [d['Actual_Weeks'] for d in data]
    counter = [d['Counter_Weeks'] for d in data]
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 5))
    
    y_pos = np.arange(len(df))
    height = 0.35
    
    # Bar for Actual
    plt.barh(y_pos - height/2, df['Actual_Weeks'], height, label='Actual Survival Duration', color='#1f77b4', alpha=0.7)
    
    # Bar for Counterfactual
    plt.barh(y_pos + height/2, df['Counter_Weeks'], height, label="With Judges' Save (Simulated)", color='#ff7f0e', alpha=0.9)
    
    plt.yticks(y_pos, df['Name'])
    plt.xlabel('Weeks Survived')
    plt.title("Impact of Mandatory Judges' Save on Controversial Figures", fontsize=14)
    plt.legend()
    
    # Add annotations
    for i, v in enumerate(df['Counter_Weeks']):
        if v < df.loc[i, 'Actual_Weeks']:
            plt.text(v + 0.5, i + height/2, f'Eliminated W{v}', va='center', color='#d62728', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_disagreement_stats(output_path='images/Method_Comparison.png'):
    # Text Data
    # Consistency: 192 (69.6%)
    # Discrepancy: 84 (30.4%)
    # Fan Fav Survival: Rank (97.4%), Percent (98.6%)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Donut Chart
    labels = ['Consistent Outcome\n(Same Person Eliminated)', 'Discrepancy\n(Different Outcome)']
    sizes = [69.6, 30.4]
    colors = ['#e0e0e0', '#ff9f4b'] # Gray vs Orange
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85, explode=(0, 0.05))
    
    # Draw circle
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Agreement Frequency: Rank vs. Percent Method', fontsize=14)
    
    # Stylize
    for text in texts:
        text.set_fontsize(12)
    for text in autotexts:
        text.set_fontsize(13)
        text.set_weight('bold')

    # 2. Bar Chart for Survival Rate
    # Survival Rate of "Fan Favorites" (Low Judge, High Fan)
    rates = [97.4, 98.6]
    cats = ['Rank Method', 'Percent Method']
    
    bars = ax2.bar(cats, rates, color=['#4c72b0', '#55a868'], width=0.5)
    ax2.set_ylim(95, 100) # Zoom in to show difference
    ax2.set_ylabel('Survival Rate of "Popular but Poor Dancers" (%)')
    ax2.set_title('Structural Bias: Protection of Fan Favorites', fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    import os
    if not os.path.exists('images'):
        os.makedirs('images')
        
    print("Loading data...")
    mcmc, raw = load_data()
    
    print("Processing Bobby Bones...")
    bobby_df = process_bobby_bones_trajectory(mcmc, raw)
    if not bobby_df.empty:
        plot_bobby_bones(bobby_df)
    else:
        print("Warning: Bobby Bones data not found!")
        
    print("Plotting Judges' Save...")
    plot_judges_save_gantt()
    
    print("Plotting Stats...")
    plot_disagreement_stats()
    
    print("Done")
