import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')
df_all = pd.get_dummies(df, columns=['State'], drop_first=True)
df_all = df_all.rename(columns={'State_Florida': 'Florida', 'State_New York': 'New York'})

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Exact rankings as in feature_selection.py
rankings = {
    'SFS (Forward)': ['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration'],
    'RFE': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York'],
    'SelectKBest': ['R&D Spend', 'Marketing Spend', 'Administration', 'New York', 'Florida'],
    'Lasso (L1)': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York'],
    'Random Forest': ['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York']
}

for algo, rank in rankings.items():
    print(f"\n--- {algo} ---")
    for k in range(1, 6):
        features = rank[:k]
        lr = LinearRegression()
        lr.fit(X_train[features], y_train)
        y_pred = lr.predict(X_test[features])
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"k={k}: R2={r2:.6f}, RMSE={rmse:.6f}")
