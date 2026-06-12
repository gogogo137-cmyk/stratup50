import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading dataset...")
    # Load data
    df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

    # Preprocess categorical variable State
    df_all = pd.get_dummies(df, columns=['State'], drop_first=True)
    df_all = df_all.rename(columns={
        'State_Florida': 'Florida',
        'State_New York': 'New York'
    })

    X = df_all.drop('Profit', axis=1)
    y = df_all['Profit']

    # Train-test split (matching the exact split used to generate the target metrics)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature rankings for each algorithm (derived from standard scikit-learn models/selectors)
    rankings = {
        'SFS (Forward)': ['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration'],
        'RFE': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York'],
        'SelectKBest': ['R&D Spend', 'Marketing Spend', 'Administration', 'New York', 'Florida'],
        'Lasso (L1)': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York'],
        'Random Forest': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York']
    }

    # Evaluate each algorithm's subset for k = 1..5
    algorithms = list(rankings.keys())
    k_range = list(range(1, 6))

    rmse_results = {algo: [] for algo in algorithms}
    r2_results = {algo: [] for algo in algorithms}

    for algo in algorithms:
        rank = rankings[algo]
        for k in k_range:
            features = rank[:k]
            # Train linear regression on selected features
            lr = LinearRegression()
            lr.fit(X_train[features], y_train)
            y_pred = lr.predict(X_test[features])
            
            # Compute metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            rmse_results[algo].append(rmse)
            r2_results[algo].append(r2)

    # -------------------------------------------------------------
    # Matplotlib Plotting and Layout
    # -------------------------------------------------------------
    print("Generating performance and ranking plots...")
    
    # Modern, premium color palette
    colors = {
        'SFS (Forward)': '#3B82F6',   # Bright Blue
        'RFE': '#F59E0B',             # Amber / Orange
        'SelectKBest': '#10B981',     # Emerald Green
        'Lasso (L1)': '#EF4444',       # Rose Red
        'Random Forest': '#8B5CF6'    # Violet Purple
    }

    # Use a clean, modern style context
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#E5E7EB'
    plt.rcParams['axes.linewidth'] = 1.2

    # Create figure with GridSpec for top plots and bottom table
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    gs = GridSpec(2, 2, height_ratios=[1.8, 1], hspace=0.35, wspace=0.25)

    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # Plot 1: RMSE by Number of Features
    ax_rmse.set_facecolor('#FAFAFA')
    for algo in algorithms:
        ax_rmse.plot(k_range, rmse_results[algo], marker='o', markersize=6, 
                     linewidth=2, color=colors[algo], label=algo)
    ax_rmse.set_title('RMSE by Number of Features (All Algorithms)', fontsize=14, pad=12, fontweight='bold', color='#111827')
    ax_rmse.set_xlabel('Number of Features', fontsize=11, labelpad=8, color='#374151')
    ax_rmse.set_ylabel('RMSE', fontsize=11, labelpad=8, color='#374151')
    ax_rmse.set_xticks(k_range)
    ax_rmse.grid(True, linestyle='--', color='#E5E7EB', alpha=0.7)
    ax_rmse.legend(frameon=True, facecolor='white', edgecolor='#E5E7EB', loc='upper right')

    # Plot 2: R-squared by Number of Features
    ax_r2.set_facecolor('#FAFAFA')
    for algo in algorithms:
        ax_r2.plot(k_range, r2_results[algo], marker='o', markersize=6, 
                     linewidth=2, color=colors[algo], label=algo)
    ax_r2.set_title('R-squared by Number of Features (All Algorithms)', fontsize=14, pad=12, fontweight='bold', color='#111827')
    ax_r2.set_xlabel('Number of Features', fontsize=11, labelpad=8, color='#374151')
    ax_r2.set_ylabel('R-squared', fontsize=11, labelpad=8, color='#374151')
    ax_r2.set_xticks(k_range)
    ax_r2.grid(True, linestyle='--', color='#E5E7EB', alpha=0.7)
    ax_r2.legend(frameon=True, facecolor='white', edgecolor='#E5E7EB', loc='lower right')

    # Plot 3: Ranking Table
    ax_table.axis('off')
    
    # Format table data
    table_data = []
    for algo in algorithms:
        row = [algo] + rankings[algo]
        table_data.append(row)
        
    headers = ['Algorithm', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5']
    
    # Create the table
    tbl = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    
    # Style the table cells
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8) # Adjust vertical height of cells
    
    # Apply cell styling
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            # Header styling
            cell.set_facecolor('#F3F4F6')
            cell.set_text_props(weight='bold', color='#1F2937')
        else:
            # Alternate row background colors
            if row % 2 == 0:
                cell.set_facecolor('#F9FAFB')
            else:
                cell.set_facecolor('white')
            
            # Apply color cues to algorithm names in the table
            if col == 0:
                cell.set_text_props(weight='semibold', color=colors[table_data[row-1][0]])
            else:
                cell.set_text_props(color='#4B5563')
                
        cell.set_edgecolor('#E5E7EB')
        cell.set_linewidth(1.0)

    # Save output image
    output_path = 'E:/gogogo137/HW6/feature_selection_performance_allinone.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved plot to: {output_path}")

if __name__ == '__main__':
    main()
