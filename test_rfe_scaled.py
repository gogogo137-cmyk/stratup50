import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

df = pd.read_csv('E:/gogogo137/HW6/50_Startups.csv')

# Preprocess
df_all = pd.get_dummies(df, columns=['State'], drop_first=True)
df_all = df_all.rename(columns={
    'State_Florida': 'Florida',
    'State_New York': 'New York'
})

X = df_all.drop('Profit', axis=1)
y = df_all['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

rfe = RFE(LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_scaled, y_train)
rfe_ranking = sorted(zip(rfe.ranking_, X.columns))
print("RFE Ranking (scaled):", rfe_ranking)
