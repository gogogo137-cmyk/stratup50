import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

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

# 1. RFE
rfe = RFE(LinearRegression(), n_features_to_select=1)
rfe.fit(X_train, y_train)
# The ranking_ attribute gives the rank. Rank 1 features are selected.
# Let's sort features by RFE ranking:
rfe_ranking = sorted(zip(rfe.ranking_, X.columns))
print("RFE Ranking:", rfe_ranking)

# 2. SelectKBest
skb = SelectKBest(f_regression, k='all')
skb.fit(X_train, y_train)
# Sort by score in descending order
skb_ranking = sorted(zip(skb.scores_, X.columns), reverse=True)
print("SelectKBest Ranking:", skb_ranking)

# 3. Lasso
# Need to scale features first for Lasso coefficient comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lasso = Lasso(alpha=100.0) # Choose some alpha
lasso.fit(X_train_scaled, y_train)
lasso_ranking = sorted(zip(np.abs(lasso.coef_), X.columns), reverse=True)
print("Lasso Ranking:", lasso_ranking)

# 4. Random Forest
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)
rf_ranking = sorted(zip(rf.feature_importances_, X.columns), reverse=True)
print("Random Forest Ranking:", rf_ranking)
