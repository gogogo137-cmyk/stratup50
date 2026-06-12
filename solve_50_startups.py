import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import joblib

def main():
    print("=" * 80)
    print("CRISP-DM Process: Solving the 50 Startups Profit Prediction Problem (v2)")
    print("=" * 80)

    # ---------------------------------------------------------
    # PHASE 1: BUSINESS UNDERSTANDING
    # ---------------------------------------------------------
    print("\n--- PHASE 1: BUSINESS UNDERSTANDING ---")
    print("Goal: Predict startup profitability based on expenses and location.")
    print("Features:")
    print("  - R&D Spend, Administration, Marketing Spend (Numerical)")
    print("  - State (Categorical)")
    print("Target:")
    print("  - Profit (Numerical)")
    print("Objective: Investigate factors driving profit and optimize regression diagnostics.")

    # ---------------------------------------------------------
    # PHASE 2: DATA UNDERSTANDING
    # ---------------------------------------------------------
    print("\n--- PHASE 2: DATA UNDERSTANDING ---")
    csv_path = os.path.join(os.path.dirname(__file__), "50_Startups.csv")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # 1. Multicollinearity Diagnostic (VIF calculation)
    print("\nCalculating Variance Inflation Factor (VIF) for Collinearity Diagnostics:")
    X_num = df[['R&D Spend', 'Administration', 'Marketing Spend']]
    X_vif = add_constant(X_num)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    # Exclude constant from printing
    print(vif_data[vif_data["Feature"] != "const"].to_string(index=False))
    print("Note: VIF < 5 indicates low/moderate multicollinearity, while VIF > 5-10 indicates high collinearity.")

    # 2. Visualizing Correlations (Heatmap)
    print("\nGenerating Correlation Matrix...")
    numerical_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    corr_matrix = df[numerical_cols].corr()
    print("Correlation matrix:")
    print(corr_matrix)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    heatmap_path = os.path.join(os.path.dirname(__file__), "correlation_matrix.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")

    # 3. Visualizing Outliers (Boxplot)
    print("\nGenerating Boxplot for Outlier Detection...")
    plt.figure(figsize=(10, 6))
    df_box = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']]
    sns.boxplot(data=df_box, palette="Set3")
    plt.title("Boxplot of Continuous Variables for Outlier Detection")
    plt.ylabel("Amount ($)")
    plt.tight_layout()
    boxplot_path = os.path.join(os.path.dirname(__file__), "outliers_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Outliers boxplot saved to: {boxplot_path}")

    # ---------------------------------------------------------
    # PHASE 3: DATA PREPARATION
    # ---------------------------------------------------------
    print("\n--- PHASE 3: DATA PREPARATION ---")
    
    # 1. Outlier Identification and Removal (IQR Method)
    Q1 = df['Profit'].quantile(0.25)
    Q3 = df['Profit'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Profit'] < lower_bound) | (df['Profit'] > upper_bound)]
    print(f"IQR Bounds for Target (Profit): Lower = ${lower_bound:.2f}, Upper = ${upper_bound:.2f}")
    if len(outliers) > 0:
        print(f"Identified {len(outliers)} outlier(s) to remove:")
        print(outliers)
        df_cleaned = df[(df['Profit'] >= lower_bound) & (df['Profit'] <= upper_bound)].copy()
    else:
        print("No outliers identified in the target variable.")
        df_cleaned = df.copy()

    # Preprocessing Pipeline Configuration
    numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    categorical_features = ['State']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # ---------------------------------------------------------
    # PHASE 4 & 5: MODELING & EVALUATION (Raw vs. Cleaned Data)
    # ---------------------------------------------------------
    print("\n--- PHASE 4 & 5: MODELING & EVALUATION ---")
    
    datasets = {
        "Raw Dataset": df,
        "Cleaned Dataset (Outliers Removed)": df_cleaned
    }
    
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    all_results = []
    trained_pipelines = {}
    best_overall_r2 = -1
    best_overall_pipeline = None
    best_overall_name = ""

    # Plot Setup: Comparing Predictions of best model on both datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ds_name, data) in enumerate(datasets.items()):
        X = data.drop('Profit', axis=1)
        y = data['Profit']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nTraining on {ds_name} (Train={len(X_train)}, Test={len(X_test)}):")
        
        ds_predictions = {}
        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            ds_predictions[model_name] = (y_test, y_pred)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            all_results.append({
                "Dataset": ds_name,
                "Model": model_name,
                "R2 Score": r2,
                "MAE": mae,
                "RMSE": rmse
            })
            
            # Keep track of best overall model to deploy
            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_pipeline = pipeline
                best_overall_name = f"{model_name} ({ds_name})"
                
        # Draw actual vs predicted for Random Forest on this dataset
        rf_y_test, rf_y_pred = ds_predictions["Random Forest Regressor"]
        axes[ax_idx].scatter(rf_y_test, rf_y_pred, color='blue', edgecolors='k', alpha=0.7)
        axes[ax_idx].plot([rf_y_test.min(), rf_y_test.max()], [rf_y_test.min(), rf_y_test.max()], 'r--', lw=2)
        axes[ax_idx].set_xlabel('Actual Profit')
        axes[ax_idx].set_ylabel('Predicted Profit')
        axes[ax_idx].set_title(f'RF Actual vs Predicted ({ds_name})')
        axes[ax_idx].grid(True, linestyle=':', alpha=0.6)

    # Save Comparison Plots
    plt.tight_layout()
    comparison_path = os.path.join(os.path.dirname(__file__), "model_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"\nModel prediction comparison plot saved to: {comparison_path}")

    # Display results comparison table
    results_df = pd.DataFrame(all_results)
    print("\nSummary Comparison Table (Raw vs. Cleaned):")
    print(results_df.to_string(index=False))

    print(f"\nOverall Winner: {best_overall_name} with R2 Score of {best_overall_r2:.4f}")

    # Feature Importance of Best Model
    print("\nFeature Impact Analysis (Best Model):")
    regressor = best_overall_pipeline.named_steps['regressor']
    encoder = best_overall_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cats = list(encoder.get_feature_names_out(['State']))
    feature_names = numeric_features + encoded_cats
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        print(imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))
    elif hasattr(regressor, 'coef_'):
        coefs = regressor.coef_
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        print(coef_df.to_string(index=False))

    # ---------------------------------------------------------
    # PHASE 6: DEPLOYMENT
    # ---------------------------------------------------------
    print("\n--- PHASE 6: DEPLOYMENT ---")
    model_export_path = os.path.join(os.path.dirname(__file__), "best_startup_model.joblib")
    joblib.dump(best_overall_pipeline, model_export_path)
    print(f"Saved the best pipeline ({best_overall_name}) to: {model_export_path}")
    
    # Demonstration of loading the model and making a prediction
    print("\nDeploying prediction test...")
    loaded_pipeline = joblib.load(model_export_path)
    
    sample_startups = pd.DataFrame([
        {
            'R&D Spend': 120000.0,
            'Administration': 90000.0,
            'Marketing Spend': 300000.0,
            'State': 'New York'
        },
        {
            'R&D Spend': 50000.0,
            'Administration': 130000.0,
            'Marketing Spend': 100000.0,
            'State': 'California'
        }
    ])
    
    predictions_demo = loaded_pipeline.predict(sample_startups)
    for idx, row in sample_startups.iterrows():
        print(f"Startup {idx+1}: Location={row['State']}, R&D={row['R&D Spend']:.2f}, Admin={row['Administration']:.2f}, Marketing={row['Marketing Spend']:.2f}")
        print(f"  --> Predicted Profit: ${predictions_demo[idx]:,.2f}")
    
    print("\n" + "=" * 80)
    print("CRISP-DM Execution Finished Successfully with Step 2 & 3 Enhancements!")
    print("=" * 80)

if __name__ == "__main__":
    main()
