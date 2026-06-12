import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

state_dummies = pd.get_dummies(df['State'], prefix='State', dtype=float)
df_all = pd.concat([df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']], state_dummies], axis=1)

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Run SFS for k=1..6
for k in range(1, 7):
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=5)
    sfs.fit(X_train, y_train)
    selected_features = list(X.columns[sfs.get_support()])
    
    # Evaluate on test set
    lr = LinearRegression()
    lr.fit(X_train[selected_features], y_train)
    y_pred = lr.predict(X_test[selected_features])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"k={k} selected: {selected_features} -> R2: {r2:.6f}, RMSE: {rmse:.6f}")
