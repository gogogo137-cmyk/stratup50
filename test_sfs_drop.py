import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

# Dummy encode State with drop_first=True
df_all = pd.get_dummies(df, columns=['State'], drop_first=True)
# Rename columns for nicer printing
df_all = df_all.rename(columns={
    'State_Florida': 'Florida',
    'State_New York': 'New York'
})

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

print("Features of X:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

for cv_val in [None, 2, 3, 4, 5]:
    print(f"\n--- SFS with cv={cv_val} ---")
    for k in range(1, 6):
        try:
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=cv_val)
            sfs.fit(X_train, y_train)
            selected_features = list(X.columns[sfs.get_support()])
            lr = LinearRegression()
            lr.fit(X_train[selected_features], y_train)
            y_pred = lr.predict(X_test[selected_features])
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"k={k} selected: {selected_features} -> R2: {r2:.6f}, RMSE: {rmse:.6f}")
        except Exception as e:
            print(f"k={k} failed: {e}")
