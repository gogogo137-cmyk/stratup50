import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

# Dummy encode State
state_dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']], state_dummies], axis=1)

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

all_cols = list(X.columns)
print("All features:", all_cols)

for k in range(1, len(all_cols) + 1):
    for combo in combinations(all_cols, k):
        features = list(combo)
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]
        lr = LinearRegression()
        lr.fit(X_train_sub, y_train)
        y_pred = lr.predict(X_test_sub)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Check if R2 is approx 0.934707 and RMSE is approx 9137.990153
        if abs(r2 - 0.934707) < 1e-4 and abs(rmse - 9137.990153) < 1.0:
            print(f"FOUND: features={features}, R2={r2:.6f}, RMSE={rmse:.6f}")
