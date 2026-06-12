import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

# Dummy encode State without dropping first
state_dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_custom = pd.concat([df[['R&D Spend', 'Marketing Spend', 'Profit']], state_dummies], axis=1)

X = df_custom.drop('Profit', axis=1)
y = df_custom['Profit']

print("Features:")
print(X.columns)

# Split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features_to_test = [
    ['R&D Spend'],
    ['R&D Spend', 'Marketing Spend'],
    ['R&D Spend', 'Marketing Spend', 'State_New York'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
    ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'State_California']
]

for idx, features in enumerate(features_to_test):
    X_train_sub = X_train[features]
    X_test_sub = X_test[features]
    lr = LinearRegression()
    lr.fit(X_train_sub, y_train)
    y_pred = lr.predict(X_test_sub)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{idx+1} features: {features} -> R2: {r2:.6f}, RMSE: {rmse:.6f}")
