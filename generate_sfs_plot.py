import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector

# 1. Load data
csv_path = '50_Startups.csv'
df = pd.read_csv(csv_path)

# 2. Dummy encoding for State (no drop_first to match teacher's State_California in k=5)
state_dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']], state_dummies], axis=1)

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

# 3. Train-test split (random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Run SFS and evaluate
k_range = list(range(1, 6))
rmse_results = []
r2_results = []
selected_features_list = []

for k in k_range:
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=5)
    sfs.fit(X_train, y_train)
    selected = list(X.columns[sfs.get_support()])
    
    # Sort selected features to match teacher's exact column order
    display_order = ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'State_California', 'Administration']
    selected_sorted = sorted(selected, key=lambda f: display_order.index(f) if f in display_order else 99)
    selected_features_list.append(str(selected_sorted))
    
    # Evaluate
    lr = LinearRegression()
    lr.fit(X_train[selected], y_train)
    y_pred = lr.predict(X_test[selected])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    rmse_results.append(rmse)
    r2_results.append(r2)

# 5. Plotting
fig = plt.figure(figsize=(10, 6.5), facecolor='white')
gs = GridSpec(2, 2, height_ratios=[1.8, 1], hspace=0.35, wspace=0.25)

ax_rmse = fig.add_subplot(gs[0, 0])
ax_r2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

# Plot RMSE
ax_rmse.plot(k_range, rmse_results, marker='o', color='#1f77b4', linewidth=1.2, markersize=4)
ax_rmse.set_title('RMSE by Number of Features', fontsize=10, pad=8)
ax_rmse.set_xlabel('Number of Features', fontsize=9)
ax_rmse.set_ylabel('RMSE', fontsize=9)
ax_rmse.set_xticks(np.arange(1.0, 5.5, 0.5))
ax_rmse.set_xlim(0.8, 5.2)
ax_rmse.tick_params(axis='both', which='major', labelsize=8)

# Plot R-squared
ax_r2.plot(k_range, r2_results, marker='o', color='#1f77b4', linewidth=1.2, markersize=4)
ax_r2.set_xlabel('Number of Features', fontsize=9)
ax_r2.set_ylabel('R-squared', fontsize=9)
ax_r2.set_xticks(np.arange(1.0, 5.5, 0.5))
ax_r2.set_xlim(0.8, 5.2)
ax_r2.tick_params(axis='both', which='major', labelsize=8)

# Draw Table
ax_table.axis('off')

# Format table data
table_data = []
for idx, k in enumerate(k_range):
    rmse_str = f"{rmse_results[idx]:.6f}"
    r2_str = f"{r2_results[idx]:.6f}"
    table_data.append([
        str(k),
        selected_features_list[idx],
        rmse_str,
        r2_str
    ])

headers = ['Number of Features', 'Selected Features', 'RMSE', 'R-squared']
tbl = ax_table.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='left')

# Style table
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.4)

# Apply alignments and borders
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
    if col == 2 or col == 3:
        cell.set_text_props(ha='right') # Right align numbers
    cell.set_edgecolor('#e0e0e0')
    cell.set_linewidth(0.8)

# Save
output_path = 'feature_selection_sfs_only.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print("Teacher-style SFS plot generated and saved successfully!")
