import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# Set page config
st.set_page_config(page_title="Startup Profit Prediction Dashboard", layout="wide")

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0b0e14;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1a1b26, #0f101a);
        color: #f8f9fa;
    }
    div[data-testid="stSidebar"] {
        background-color: #111222;
        border-right: 1px solid #1f2035;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    .card {
        background-color: #16172a;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #23243f;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 14px;
        text-align: left;
    }
    .custom-table th {
        background-color: #1e293b;
        color: #f8f9fa;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 2px solid #334155;
    }
    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #1e293b;
        color: #e2e8f0;
    }
    .custom-table tr:hover {
        background-color: #0f172a;
    }
    .feature-badge {
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px;
        border: 1px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('50_Startups.csv')
    return df

df_raw = load_data()

# Preprocess categorical variable State (One-Hot Encoding, drop_first=True to avoid dummy variable trap)
df_encoded = pd.get_dummies(df_raw, columns=['State'], drop_first=True)
df_encoded = df_encoded.rename(columns={
    'State_Florida': 'State_Florida',
    'State_New York': 'State_New York'
})

# Identify columns
X_cols = [col for col in df_encoded.columns if col != 'Profit']
y_col = 'Profit'

# 2. Sidebar Settings
st.sidebar.markdown("### ⚙️ Split & Selection Settings")
test_size_pct = st.sidebar.slider("Test Split Size (%)", min_value=10, max_value=50, value=20, step=5)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=9999)
k_features = st.sidebar.slider("Target Number of Features (k)", min_value=1, max_value=4, value=3, step=1)

st.sidebar.markdown("### 🤖 Model Architecture Selection")
st.sidebar.write("(Integrates with future regression modeling lessons)")
evaluator_model_name = st.sidebar.selectbox("Choose Evaluator Model:", ["Linear Regression", "Random Forest", "Lasso", "Ridge"])

# Train-test split
X = df_encoded[X_cols]
y = df_encoded[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100.0, random_state=random_seed)

# Map model selection
def get_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_seed)
    elif name == "Lasso":
        return Lasso(alpha=1.0)
    elif name == "Ridge":
        return Ridge(alpha=1.0)
    return LinearRegression()

evaluator_model = get_model(evaluator_model_name)

# 3. Define the 9 Feature Selection Methods
def run_feature_selection(X_tr, y_tr, k):
    methods = {}
    
    # 1. Pearson Correlation
    corrs = X_tr.apply(lambda col: y_tr.corr(col))
    methods['Pearson Correlation'] = list(corrs.abs().nlargest(k).index)
    
    # 2. Spearman Correlation
    spearman_ranks = X_tr.apply(lambda col: spearmanr(col, y_tr)[0])
    methods['Spearman Correlation'] = list(spearman_ranks.abs().nlargest(k).index)
    
    # 3. F-test Regression
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_tr, y_tr)
    methods['F-test Regression'] = list(X_tr.columns[f_selector.get_support()])
    
    # 4. Mutual Information (deterministic via random_state)
    mi_selector = SelectKBest(lambda X, y: mutual_info_regression(X, y, random_state=random_seed), k=k)
    mi_selector.fit(X_tr, y_tr)
    methods['Mutual Information'] = list(X_tr.columns[mi_selector.get_support()])
    
    # 5. Recursive Feature Elimination (RFE) - Scale features first for correct coefficients ranking
    scaler_rfe = StandardScaler()
    X_tr_scaled_rfe = pd.DataFrame(scaler_rfe.fit_transform(X_tr), columns=X_tr.columns)
    rfe_selector = RFE(LinearRegression(), n_features_to_select=k)
    rfe_selector.fit(X_tr_scaled_rfe, y_tr)
    methods['Recursive Feature Elimination (RFE)'] = list(X_tr.columns[rfe_selector.get_support()])
    
    # 6. SFS Forward
    sfs_selector = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=min(3, len(X_tr)))
    sfs_selector.fit(X_tr, y_tr)
    methods['SFS (Forward Selection)'] = list(X_tr.columns[sfs_selector.get_support()])
    
    # 7. Lasso (L1 Regularization)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    lasso = Lasso(alpha=100.0, max_iter=10000)
    lasso.fit(X_tr_scaled, y_tr)
    lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X_tr.columns)
    methods['Lasso (L1 Regularization)'] = list(lasso_coefs.nlargest(k).index)
    
    # 8. Random Forest Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=random_seed)
    rf.fit(X_tr, y_tr)
    rf_importances = pd.Series(rf.feature_importances_, index=X_tr.columns)
    methods['Random Forest Regressor'] = list(rf_importances.nlargest(k).index)
    
    # 9. Gradient Boosting Importance
    gb = GradientBoostingRegressor(random_state=random_seed)
    gb.fit(X_tr, y_tr)
    gb_importances = pd.Series(gb.feature_importances_, index=X_tr.columns)
    methods['Gradient Boosting Regressor'] = list(gb_importances.nlargest(k).index)
    
    return methods

feature_selections = run_feature_selection(X_train, y_train, k_features)

# Title
st.markdown("<h1 style='text-align: center; color: #60a5fa; margin-bottom: 0.5rem;'>🚀 Startup 50 Profit Prediction & Feature Selection</h1>", unsafe_allow_html=True)
st.write("---")

# Header Section matching Notion
st.markdown("""
<div class="card" style="border-left: 5px solid #f59e0b;">
    <h3 style="margin-top: 0; color: #f59e0b; font-size: 20px;">📊 Performance Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Main layout: Plots side-by-side
col_left, col_right = st.columns(2)

# Left column: Correlation Heatmap (with White background)
with col_left:
    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5), facecolor='white')
    ax_corr.set_facecolor('white')
    corr = df_encoded.corr()
    im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Style labels and ticks
    ax_corr.set_xticks(np.arange(len(corr.columns)))
    ax_corr.set_yticks(np.arange(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, fontsize=8, rotation=45, ha='right', color='black')
    ax_corr.set_yticklabels(corr.columns, fontsize=8, color='black')
    ax_corr.set_title("Correlation Matrix Heatmap", color='black', fontsize=11, fontweight='bold', pad=10)
    
    # Add values inside heatmap
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax_corr.text(j, i, f"{val:.2f}",
                         ha="center", va="center", color=text_color, fontsize=8)
            
    fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)

# Right column: Feature Selection Consensus (with White background)
with col_right:
    # Calculate counts
    all_selected_features = []
    for f_list in feature_selections.values():
        all_selected_features.extend(f_list)
        
    counts = pd.Series(all_selected_features).value_counts().reindex(X_cols, fill_value=0).sort_values(ascending=True)
    
    fig_cons, ax_cons = plt.subplots(figsize=(6.5, 4.5), facecolor='white')
    ax_cons.set_facecolor('white')
    
    bars = ax_cons.barh(counts.index, counts.values, color='#3b82f6', height=0.6)
    
    # Styling labels and grid
    ax_cons.set_title('Feature Selection Consensus (Out of 9 Algorithms)', fontsize=11, fontweight='bold', color='black', pad=10)
    ax_cons.set_xlabel('Number of Algorithms Selecting Feature', fontsize=9, color='black', labelpad=5)
    ax_cons.set_xticks(range(10))
    ax_cons.set_xticklabels(range(10), color='black')
    ax_cons.set_yticklabels(counts.index, color='black', fontsize=9)
    ax_cons.grid(True, axis='x', linestyle='--', alpha=0.3, color='grey')
    
    # Add count labels next to bars
    for bar in bars:
        width = bar.get_width()
        ax_cons.text(width + 0.15, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center', color='black', fontweight='bold', fontsize=8)
        
    fig_cons.tight_layout()
    st.pyplot(fig_cons)

st.write("---")

# Tabs for detailed analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Complete Comparison Table", 
    "🔍 Detailed Method Inspector", 
    "🔮 Live Profit Simulator", 
    "📈 Stepwise Curves (All-in-One)"
])

# Tab 1: Complete Comparison Table
with tab1:
    st.markdown(f"""
    <div class="card" style="border-left: 5px solid #3b82f6; margin-bottom: 1rem;">
        <h3 style="margin-top: 0; margin-bottom: 0; color: #f8f9fa;">📋 Performance Metrics Comparison (Evaluated via {evaluator_model_name})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    comparison_data = []
    best_r2 = -999.0
    best_method = ""
    best_features = []
    
    for method, features in feature_selections.items():
        # Train and predict
        model = get_model(evaluator_model_name)
        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        comparison_data.append({
            'Method': method,
            'Selected Features': ", ".join(features),
            'Number of Features': len(features),
            'MAE': f"${mae:,.2f}",
            'RMSE': f"${rmse:,.2f}",
            'R2': f"{r2:.5f}"
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_method = method
            best_features = features
            
    # Render table in HTML/CSS with pill badges
    html_table = f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th>Method</th>
                <th>Selected Features</th>
                <th style="text-align: center;">Number of Features</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R2</th>
            </tr>
        </thead>
        <tbody>
    """
    for row in comparison_data:
        features_list = row['Selected Features'].split(", ")
        badges_html = "".join([f'<span class="feature-badge">{f}</span>' for f in features_list])
        
        html_table += f"""
            <tr>
                <td style="font-weight: bold; color: #60a5fa;">{row['Method']}</td>
                <td>{badges_html}</td>
                <td style="text-align: center; font-weight: bold; color: #f8f9fa;">{row['Number of Features']}</td>
                <td style="font-family: monospace; color: #34d399; font-weight: 600;">{row['MAE']}</td>
                <td style="font-family: monospace; color: #f87171; font-weight: 600;">{row['RMSE']}</td>
                <td style="font-family: monospace; color: #facc15; font-weight: bold;">{row['R2']}</td>
            </tr>
        """
    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)
    st.write("")
    
    st.success(f"🏆 **Best Performing Selection Method**: `{best_method}` with an R2 score of **{best_r2:.5f}** using features `{best_features}`")

# Tab 2: Detailed Method Inspector
with tab2:
    st.markdown("### 🔍 Detailed Method Inspector")
    selected_inspect_method = st.selectbox("Select Feature Selection Method to Inspect:", list(feature_selections.keys()))
    
    inspect_features = feature_selections[selected_inspect_method]
    
    # Fit model on inspected features
    inspect_model = get_model(evaluator_model_name)
    inspect_model.fit(X_train[inspect_features], y_train)
    
    st.write(f"**Selected Features:** `{inspect_features}`")
    
    # If Linear Regression, display formulas
    if evaluator_model_name in ["Linear Regression", "Lasso", "Ridge"]:
        coef_df = pd.DataFrame({
            'Feature': inspect_features,
            'Coefficient (Weight)': inspect_model.coef_
        })
        st.write(coef_df)
        st.info(f"**Intercept (Base Value):** ${inspect_model.intercept_:,.2f}")
    else:
        st.write("Model parameters are configured automatically.")

# Tab 3: Live Profit Simulator
with tab3:
    st.markdown("### 🔮 Live Profit Simulator")
    st.write(f"Using the best feature selection method (`{best_method}`) trained on: `{best_features}`")
    
    # Input forms
    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    with col_in1:
        input_rd = st.number_input("R&D Spend ($)", value=100000.0, step=5000.0)
    with col_in2:
        input_admin = st.number_input("Administration ($)", value=120000.0, step=5000.0)
    with col_in3:
        input_market = st.number_input("Marketing Spend ($)", value=250000.0, step=5000.0)
    with col_in4:
        input_state = st.selectbox("State:", df_raw['State'].unique())
        
    # Map input to dummy variables
    input_data = pd.DataFrame([{
        'R&D Spend': input_rd,
        'Administration': input_admin,
        'Marketing Spend': input_market,
        'State_Florida': 1.0 if input_state == 'Florida' else 0.0,
        'State_New York': 1.0 if input_state == 'New York' else 0.0
    }])
    
    # Extract only the selected best features
    input_features_data = input_data[best_features]
    
    # Fit the best model on full training set of best features
    best_model_deploy = get_model(evaluator_model_name)
    best_model_deploy.fit(X_train[best_features], y_train)
    predicted_val = best_model_deploy.predict(input_features_data)[0]
    
    st.markdown(f"## 💵 Predicted Profit: <span style='color: #10b981;'>${predicted_val:,.2f}</span>", unsafe_allow_html=True)

# Tab 4: Stepwise Curves (All-in-One)
with tab4:
    st.markdown("### 📈 Stepwise Curves (R2 and RMSE as k ranges from 1 to 5)")
    
    k_vals = list(range(1, 6))
    
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # SFS Forward Selection logic for k=1..5
    rmse_curve = []
    r2_curve = []
    
    for k in k_vals:
        sfs_c = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=3)
        sfs_c.fit(X_train, y_train)
        sel_c = list(X.columns[sfs_c.get_support()])
        
        lr_c = LinearRegression()
        lr_c.fit(X_train[sel_c], y_train)
        y_pred_c = lr_c.predict(X_test[sel_c])
        
        r2_curve.append(r2_score(y_test, y_pred_c))
        rmse_curve.append(np.sqrt(mean_squared_error(y_test, y_pred_c)))
        
    # Plot RMSE
    ax1.plot(k_vals, rmse_curve, marker='o', color='#3b82f6', linewidth=2, label="SFS (Forward)")
    ax1.set_title('RMSE by Number of Features', color='black', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Features', color='black', fontsize=9)
    ax1.set_ylabel('RMSE', color='black', fontsize=9)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle='--', alpha=0.3, color='grey')
    ax1.tick_params(colors='black', labelsize=8)
    
    # Plot R2
    ax2.plot(k_vals, r2_curve, marker='o', color='#10b981', linewidth=2, label="SFS (Forward)")
    ax2.set_title('R-squared by Number of Features', color='black', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Features', color='black', fontsize=9)
    ax2.set_ylabel('R-squared', color='black', fontsize=9)
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle='--', alpha=0.3, color='grey')
    ax2.tick_params(colors='black', labelsize=8)
    
    fig_curves.tight_layout()
    st.pyplot(fig_curves)

# 4. Chat History Log Section
st.write("---")
st.markdown("## 💬 我與IDE的對話LOG")
try:
    with open('chat_history.md', 'r', encoding='utf-8') as chat_f:
        chat_content = chat_f.read()
    with st.expander("🔍 點擊展開查看完整對話歷程 (Click to expand and view complete chat history)", expanded=False):
        st.markdown(chat_content)
except Exception as e:
    st.error(f"無法載入對話紀錄：{e}")

