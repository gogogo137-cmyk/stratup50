import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')
df_all = pd.get_dummies(df, columns=['State'], drop_first=True)
df_all = df_all.rename(columns={'State_Florida': 'Florida', 'State_New York': 'New York'})

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

combinations_step3 = [
    ['R&D Spend', 'Marketing Spend', 'New York'],
    ['R&D Spend', 'Marketing Spend', 'Florida'],
    ['R&D Spend', 'Marketing Spend', 'Administration']
]

for combo in combinations_step3:
    lr = LinearRegression()
    lr.fit(X_train[combo], y_train)
    train_r2 = r2_score(y_train, lr.predict(X_train[combo]))
    test_r2 = r2_score(y_test, lr.predict(X_test[combo]))
    print(f"Combo: {combo} -> Train R2: {train_r2:.6f}, Test R2: {test_r2:.6f}")
