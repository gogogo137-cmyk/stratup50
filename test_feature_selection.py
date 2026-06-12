import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

# Handle categorical variables (One-hot encode State)
# We want: California, Florida, New York
# In pandas, get_dummies or OneHotEncoder
# Let's use OneHotEncoder(drop='first') or get_dummies(drop_first=True)
# If drop='California', dummy features will be State_Florida, State_New York
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
print("Columns after dummy encoding:")
print(df_encoded.columns)

# Try with and without outlier removal
# Outlier removal:
Q1 = df['Profit'].quantile(0.25)
Q3 = df['Profit'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_encoded[(df_encoded['Profit'] >= lower_bound) & (df_encoded['Profit'] <= upper_bound)].copy()

datasets = {
    "raw": df_encoded,
    "cleaned": df_cleaned
}

for name, data in datasets.items():
    X = data.drop('Profit', axis=1)
    y = data['Profit']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SFS Forward
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=1, direction='forward')
    sfs.fit(X_train, y_train)
    
    # Let's run a test for linear regression with R&D Spend only on test set
    X_train_sub = X_train[['R&D Spend']]
    X_test_sub = X_test[['R&D Spend']]
    lr = LinearRegression()
    lr.fit(X_train_sub, y_train)
    y_pred = lr.predict(X_test_sub)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Dataset: {name} | 1 feature (R&D Spend) -> R2: {r2:.6f}, RMSE: {rmse:.6f}")
