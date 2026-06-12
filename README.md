# L9 Kaggle 50 HW6 - CRISP-DM 專案分析報告

本報告完整記錄了基於 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 流程，針對 **Kaggle 50 Startups** 資料集進行新創公司利潤預測與特徵篩選的機器學習專案開發歷程與技術手冊。

---

## 📌 專案目錄與 CRISP-DM 流程架構
1. **[1. Business Understanding](#1-business-understanding)**
2. **[2. Data Understanding](#2-data-understanding)**
3. **[3. Data Preprocessing/A coding draft v1/draft2 with 專家討論](#3-data-preprocessinga-coding-draft-v1draft2-with-專家討論)**
4. **[4. Model Selection/Modeling (Multiple Linear regression)](#4-model-selectionmodeling-multiple-linear-regression)**
5. **[5. Model Evaluation](#5-model-evaluation)**
6. **[6. Deploy/Production](#6-deployproduction)**
7. **[7. Final Result](#7-final-result)**

---

## 1. Business Understanding

新創公司（Startups）的成功與否與其資金分配有著極高關聯。創投機構（VC）或企業決策者在進行投資評估時，需要了解各項支出（研發、行政、行銷）如何影響公司的最終利潤，藉此優化投資組合與資源配置。

### 核心商業問題
> **如何根據新創公司的不同支出項目及所在地區，精準預測該公司的最終利潤（Profit）？**

### 關鍵商業問題 (Key Business Questions)
1. **更高的 R&D Spend（研發支出）是否值必定帶來更高的利潤？**
2. **Marketing Spend（行銷支出）與利潤之間是否存在強烈正相關？**
3. **Administration Spend（行政管理支出）是否會顯著影響利潤的產生？**
4. **新創公司所在的州別（State）會如何影響其盈利能力？**
5. **我們能否建立一個高精準度的迴歸模型，用以輔助創投進行利潤預測？**

### 專案目標 (Project Goal)
本專案旨在開發一個基於 `scikit-learn` 的監督式機器學習迴歸模型，輸入新創公司的研發、行政、行銷支出及所在州別，輸出其預測利潤（Profit），並深入分析各特徵對利潤的影響力。

---

## 2. Data Understanding

本階段重點在於探索資料集結構、欄位分布、多重共線性診斷（VIF）及進行專家特徵評估。

### 四個特徵的專家分級

在進行機器學習建模前，結合商業邏輯與統計分析，對以下 4 個輸入特徵進行專家級定位：

| 特徵 | 專家定位 | 預期重要性 | 建議 |
| :--- | :--- | :---: | :--- |
| **R&D Spend** | 核心成長因子 | 很高 | 一定保留 |
| **Marketing Spend** | 市場擴張因子 | 中高 | 保留，但注意共線性 |
| **Administration** | 營運成本 / 規模因子 | 低到中 | 先保留，後續評估 |
| **State** | 地區輔助因子 | 低到中 | One-Hot 後保留，謹慎解讀 |

> [!WARNING]
> **關於 State（地區輔助因子）的推論限制：**
> 雖然在描述性統計中可能會發現 Florida 平均利潤較高，但絕不能直接推論出「在 Florida 創業比在 California 更容易成功」。State 變數可能捕捉到的是地區稅率或基礎建設差異，但在本資料集（僅 50 筆）中其僅能作為輔助變數，不應過度解讀因果關係。

### 特徵相關性與共線性診斷

#### 1. 共線性診斷 (VIF 計算)
為防範多重共線性（Multicollinearity）影響線性模型係數解釋力，我們計算了連續型特徵的 **變異數膨脹因子 (Variance Inflation Factor, VIF)**：
*   **R&D Spend**: 2.47
*   **Marketing Spend**: 2.33
*   **Administration**: 1.18

VIF 值均小於關鍵門檻值 5（甚至 10），顯示各支出特徵間不存在嚴重多重共線性，可安心投入線性模型。

#### 2. 特徵相關性矩陣
數值型特徵與目標變數的 Pearson 相關係數如下：
*   **R&D Spend 與 Profit**: $r = 0.97$ (極高正相關)
*   **Marketing Spend 與 Profit**: $r = 0.75$ (顯著正相關)
*   **Administration 與 Profit**: $r = 0.20$ (弱正相關)

![特徵相關性矩陣](./correlation_matrix.png)

#### 3. 離群值檢測箱形圖
箱形圖顯示，大部分欄位的分布非常健康，僅有目標變數 `Profit` 存在一個低於下邊界的極端值（Outlier）。

![離群值檢測箱形圖](./outliers_boxplot.png)

### 類別型特徵與獨熱編碼分析 (One-Hot Encoding)

在進行機器學習建模時，我們的模型無法直接讀取非數值的文字，因此需要對類別型特徵進行處理。

#### 1. 什麼是 One-Hot Encoding (獨熱編碼)？
機器學習模型（例如多元線性迴歸）本質上是數學方程式，只能處理**數字**，看不懂文字（如 `California`、`Florida`、`New York`）。
**One-Hot Encoding** 的作用就是把這些**類別文字轉換成由 0 和 1 組成的數值欄位**。每一個類別都會獨立變成一個新的欄位：
- 當資料符合該類別時，該欄位標記為 `1` (代表「是/Yes」)；
- 不符合時，標記為 `0` (代表「否/No」)。

#### 2. 在本專案中的具體運作方式
專案中的 `State` 欄位原本包含三個文字值：`California`、`Florida`、`New York`。經過 One-Hot Encoding 後，會轉化為三個布林欄位：

| 原始 State 欄位 | State_California | State_Florida | State_New York |
| :--- | :---: | :---: | :---: |
| **New York** | 0 | 0 | 1 |
| **California** | 1 | 0 | 0 |
| **Florida** | 0 | 1 | 0 |

#### 3. 防範虛擬變數陷阱 (Dummy Variable Trap)
因為 `State_California + State_Florida + State_New York` 的總和一定等於 1，這在統計學上會造成「資訊完全重複」（多重共線性），會使線性迴歸模型無法正常計算。
為了避免這個問題，我們使用 `OneHotEncoder(drop='first')` 參數**丟棄第一個類別**（在此為 `California`），只保留 `Florida` 和 `New York` 兩個特徵欄位。如果這兩個欄位皆為 0，模型自然能推導出這家公司屬於 `California`。

---

## 3. Data Preprocessing/A coding draft v1/draft2 with 專家討論

資料準備是機器學習成功的基石。我們實作了以下特徵工程流程（經由與領域專家討論，確定排除雜訊並做適當轉換）：

1.  **離群值移除 (IQR 方法 - 專家討論後決定過濾)**
    我們對目標變數 `Profit` 計算第一四分位數 ($Q_1$) 與目標變數第三四分位數 ($Q_3$)，得出 IQR。
    *   下界 (Lower Bound) = \$15,698.29
    *   上界 (Upper Bound) = \$223,482.89
    *   **識別結果**：成功篩選出 1 筆低於下界的離群值（Profit = \$14,681.40），並將其從 Cleaned Dataset 中移除，以防止模型對極端值產生偏誤。

2.  **類別變數編碼 (One-Hot Encoding - 專家強調 State 必做)**
    類別變數 `State` (包含 California, Florida, New York) 被轉換為虛擬變數 (Dummy Variables)。為了**防止虛擬變數陷阱 (Dummy Variable Trap)**，我們在建模時丟棄了第一個類別（California），僅保留 `Florida` 與 `New York` 作為特徵。

3.  **特徵標準化 (Standardization - 數值縮放草稿 v1)**
    為了確保不同尺度的數值欄位不會偏袒特定特徵，並有利於 Lasso 等正規化模型收斂，數值型欄位 (`R&D Spend`, `Administration`, `Marketing Spend`) 均採用 `StandardScaler` 標準化為平均數為 0、標準差為 1 的分布。

4.  **資料集切分 (Train-Test Split - 驗證設計草稿 v2)**
    將資料集按 **80% 訓練集** 與 **20% 測試集** 進行切分，確保模型能進行公平泛化評估。

---

## 4. Model Selection/Modeling (Multiple Linear regression)

為了找出最佳的模型架構，我們比較了多種迴歸演算法，並在此階段分析了特徵選取的敏感度。

### 特徵選擇分析 (Model Selection / Feature Selection)
我們評估了五種特徵選擇演算法在特徵個數 $k \in [1, 5]$ 時的表現：
1.  **Sequential Feature Selector (SFS - 前向選擇)**
2.  **Recursive Feature Elimination (RFE - 遞迴特徵消除)**
3.  **SelectKBest (Filter 方法，基於 F 檢定)**
4.  **Lasso (L1 係數收縮)**
5.  **Random Forest Regressor (基於不純度減少的特徵重要性)**

下圖展示了各演算法在不同特徵數量下的 **RMSE** 與 **R-squared** 變化趨勢，並在下方列出其特徵重要性排序：

![特徵篩選成效與排序](./feature_selection_performance_allinone.png)

#### 各特徵選擇演算法詳細效能數據

##### Sequential Feature Selector (SFS - 前向選擇)
| 特徵數量 ($k$) | 選擇的特徵子集 (Selected Features) | $R^2$ 評分 | 均方根誤差 (RMSE) |
| :---: | :--- | :---: | :---: |
| **k = 1** | `['R&D Spend']` | 0.9465 | $8,274.87 |
| **k = 2** | `['R&D Spend', 'Marketing Spend']` | **0.9474** | **$8,198.80** |
| **k = 3** | `['R&D Spend', 'Marketing Spend', 'New York']` | 0.9460 | $8,309.06 |
| **k = 4** | `['R&D Spend', 'Marketing Spend', 'New York', 'Florida']` | 0.9447 | $8,409.92 |
| **k = 5** | `['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration']` | 0.9347 | $9,137.99 |

##### Recursive Feature Elimination (RFE - 遞迴特徵消除)
| 特徵數量 ($k$) | 選擇的特徵子集 (Selected Features) | $R^2$ 評分 | 均方根誤差 (RMSE) |
| :---: | :--- | :---: | :---: |
| **k = 1** | `['R&D Spend']` | 0.9465 | $8,274.87 |
| **k = 2** | `['R&D Spend', 'Marketing Spend']` | **0.9474** | **$8,198.80** |
| **k = 3** | `['R&D Spend', 'Marketing Spend', 'Administration']` | 0.9394 | $8,803.78 |
| **k = 4** | `['R&D Spend', 'Marketing Spend', 'Administration', 'Florida']` | 0.9357 | $9,068.54 |
| **k = 5** | `['R&D Spend', 'Marketing Spend', 'Administration', 'Florida', 'New York']` | 0.9347 | $9,137.99 |

### 迴歸建模架構與勝出特徵 (Modeling Conclusion)
*   **最佳特徵數量**：當特徵數量為 **$k=2$**（即納入 `R&D Spend` 與 `Marketing Spend`）時，多重線性迴歸模型展現出最佳效能，此時 **R-squared 達到最高值 0.9474**，且 **RMSE 達到最低值 8,198.80**。
*   **多元線性迴歸與其他演算法對照**：我們除了建立標準的多元線性迴歸 (Multiple Linear Regression) 之外，同時也建立了決策樹與隨機森林模型，以對比線性與非線性模型的表現。

---

## 5. Model Evaluation

我們對所有模型進行了嚴謹的指標檢驗，比較「原始資料集 (Raw)」與「清理離群值資料集 (Cleaned)」的預測結果。

### 預測成效比較表

| 資料集類型 | 演算法模型 | $R^2$ 評分 | 平均絕對誤差 (MAE) | 均方根誤差 (RMSE) |
| :--- | :--- | :--- | :--- | :--- |
| **原始資料集 (Raw)** | 多元線性迴歸 | 0.8987 | \$6,961.48 | \$9,055.96 |
| | 決策樹迴歸 | 0.8359 | \$9,131.09 | \$11,527.66 |
| | 隨機森林迴歸 | 0.9147 | \$6,131.91 | \$8,310.36 |
| **清理離群值 (Cleaned)** | 多元線性迴歸 | 0.9191 | \$6,550.86 | \$8,102.92 |
| | 決策樹迴歸 | 0.8348 | \$9,881.95 | \$11,578.30 |
| | **隨機森林迴歸 (Winner)** | **0.9260** | **\$6,892.37** | **\$7,747.89** |

### 預測結果分布圖對比
以下為隨機森林模型在 **原始資料** 與 **清除離群值** 後的預測點與完美預測線（虛線）的分布對比。點越接近虛線，代表預測越精確。顯然，清除離群值顯著提升了模型對大部分資料點的預測集中度。

![模型預測成效對比](./model_comparison.png)

---

## 6. Deploy/Production

我們已將表現最佳的隨機森林迴歸流水線（含標準化與 One-hot 編碼的前處理 Pipeline）匯出為序列化模型檔案，供生產環境進行實時推論：
👉 [best_startup_model.joblib](file:///f:/gogogo137/HW6/best_startup_model.joblib)

### Python 生產環境推論範例

```python
import joblib
import pandas as pd

# 1. 載入訓練好的最佳模型 Pipeline
model_path = "best_startup_model.joblib"
loaded_pipeline = joblib.load(model_path)

# 2. 準備待預測的新創公司財務數據
new_startups = pd.DataFrame([
    {
        'R&D Spend': 120000.0,
        'Administration': 90000.0,
        'Marketing Spend': 300000.0,
        'State': 'New York'
    },
    {
        'R&D Spend': 50000.0,
        'Administration': 130000.0,
        'Marketing Spend': 100000.0,
        'State': 'California'
    }
])

# 3. 進行預測
predicted_profits = loaded_pipeline.predict(new_startups)

# 4. 輸出預測利潤結果
print("--- 新創公司利潤預測結果 ---")
for idx, row in new_startups.iterrows():
    print(f"新創公司 {idx+1} ({row['State']}):")
    print(f"  研發支出: ${row['R&D Spend']:,.2f} | 行政支出: ${row['Administration']:,.2f} | 行銷支出: ${row['Marketing Spend']:,.2f}")
    print(f"  ==> 預估最終利潤: ${predicted_profits[idx]:,.2f}\n")
```

---

## 7. Final Result

經過完整的 CRISP-DM 專案實作與評估，本專案得出以下最終結論：

1.  **關鍵利潤驅動因子**：**R&D Spend (研發支出)** 對於新創公司的利潤具有最強烈的決定性作用（相關係數 $r=0.97$，特徵重要性排名第一）。與此同時，**Marketing Spend (行銷支出)** 也是推動利潤成長的次要關鍵因子，而 **Administration (行政支出)** 及所在 **State (州別)** 對利潤的直接邊際效應則十分有限。
2.  **模型效能優勝者**：在過濾離群值後的資料集上訓練出的 **隨機森林迴歸模型 (Random Forest Regressor)** 取得了最佳的預測成效（$R^2 = 0.9260$，$\text{RMSE} = \$7,747.89$）。
3.  **特徵精簡最佳化**：特徵篩選分析證實，僅使用 `R&D Spend` 與 `Marketing Spend` 兩個特徵所建立的簡化模型能達到最佳的泛化能力，有助於降低維度並避免模型過擬合。
