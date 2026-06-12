---
name: solve-50-startups
description: Run an end-to-end CRISP-DM machine learning regression pipeline on the 50 Startups dataset using scikit-learn.
---

# Solve 50 Startups Skill

This skill implements a robust machine learning regression pipeline to predict startup profits based on their R&D Spend, Administration Spend, Marketing Spend, and operating State.

## CRISP-DM Steps Executed:
1. **Business Understanding:** Align predictions with VC investment optimization.
2. **Data Understanding:** Analyze distributions, correlations, and calculate collinearity diagnostics via **Variance Inflation Factor (VIF)**. Visualizes data using correlation heatmaps and boxplots.
3. **Data Preparation:** Encodes categorical variable `State` (avoiding the dummy variable trap), standardizes numerical features via `StandardScaler`, and automatically removes leverage outliers using the **IQR method** on the target `Profit`.
4. **Modeling:** Fits Multiple Linear Regression, Decision Tree Regressor, and Random Forest Regressor on both raw and cleaned datasets.
5. **Evaluation:** Compares metrics ($R^2$, MAE, RMSE) of all models and outputs comparative performance graphs.
6. **Deployment:** Exports the overall best pipeline to a Joblib file (`best_startup_model.joblib`) for inference.

## How to Use:
To execute the pipeline, run the Python script located in the `scripts` directory of this skill:
```bash
python solve_50_startups.py
```
