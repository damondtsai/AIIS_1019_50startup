import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. 讀取資料與前處理 (Data Loading & Preprocessing)
# ==========================================
url = "https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv"
df = pd.read_csv(url)

# 分離特徵 (X) 與目標變數 (y)
X = df.drop('Profit', axis=1)
y = df['Profit']

# 類別變數 (State) 進行 One-Hot Encoding，drop_first=True 避免共線性
X_encoded = pd.get_dummies(X, columns=['State'], drop_first=True, dtype=float)

# 拆分訓練集 (80%) 與測試集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

feature_names = list(X_encoded.columns)
print("=== 原始特徵列表 ===")
print(feature_names)
print("-" * 65)

# ==========================================
# 2. 三種特徵篩選方法 (Feature Selection Methods)
# ==========================================

# ------------------------------------------
# 方法 1：ANOVA F-test (Filter Method 過濾法)
# ------------------------------------------
f_scores, p_values = f_regression(X_train, y_train)
anova_df = pd.DataFrame({
    'Feature': feature_names,
    'F-Score': f_scores,
    'p-value': p_values
}).sort_values(by='F-Score', ascending=False).reset_index(drop=True)

print("1. [ANOVA F-test] 特徵重要性排序：")
print(anova_df.to_string())
print("-" * 65)

# ------------------------------------------
# 方法 2：RFE 遞迴特徵消除 (Wrapper Method 包裝法)
# ------------------------------------------
# 使用標準化後的特徵消除尺度影響
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_scaled, y_train)

rfe_df = pd.DataFrame({
    'Feature': feature_names,
    'RFE_Ranking': rfe.ranking_
}).sort_values(by='RFE_Ranking').reset_index(drop=True)

print("2. [RFE 遞迴特徵消除] 特徵排名 (1 為最佳)：")
print(rfe_df.to_string())
print("-" * 65)

# ------------------------------------------
# 方法 3：Lasso 回歸 (Embedded Method 嵌入法)
# ------------------------------------------
# 利用 L1 正則化自動進行特徵選取（不重要特徵係數歸零）
lasso_cv = LassoCV(cv=5, random_state=0).fit(X_train_scaled, y_train)
lasso_df = pd.DataFrame({
    'Feature': feature_names,
    'Lasso_Coef': lasso_cv.coef_,
    'Abs_Coef': np.abs(lasso_cv.coef_)
}).sort_values(by='Abs_Coef', ascending=False).reset_index(drop=True)

print(f"3. [Lasso] 特徵係數 (最佳 alpha={lasso_cv.alpha_:.2f})：")
print(lasso_df.to_string())
print("-" * 65)

# ==========================================
# 3. 特徵數量逐步評估 (1 到 5 個特徵)
# ==========================================
feature_order = ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']

num_features = []
features_str_list = []
rmse_list = []
r2_list = []

for k in range(1, len(feature_order) + 1):
    selected_features = feature_order[:k]
    
    # 建立與訓練模型
    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])
    
    # 計算評估指標
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    num_features.append(k)
    features_str_list.append("[" + ", ".join(selected_features) + "]")
    rmse_list.append(rmse)
    r2_list.append(r2)

# 建立結果彙總 DataFrame
results_df = pd.DataFrame({
    'Number of Features': num_features,
    'Selected Features': features_str_list,
    'RMSE': rmse_list,
    'R-squared': r2_list
})

print("=== 特徵數量評估結果表 ===")
print(results_df.to_string(index=False))
print("-" * 65)

# ==========================================
# 4. 繪製圖 1：折線圖 (RMSE & R-squared 趨勢)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 左圖：RMSE by Number of Features
ax1.plot(num_features, rmse_list, marker='o')
ax1.set_title('RMSE by Number of Features')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('RMSE')

# 右圖：R-squared by Number of Features
ax2.plot(num_features, r2_list, marker='o')
ax2.set_title('R-squared by Number of Features')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('R-squared')

plt.tight_layout()
plt.savefig('features_performance.png', dpi=300)
print("1. 折線圖已儲存為 features_performance.png")

# ==========================================
# 5. 繪製圖 2：成果表格圖 (Table Visualization)
# ==========================================
fig_tbl, ax_tbl = plt.subplots(figsize=(14, 2.8), dpi=300)
ax_tbl.axis('off')

# 格式化表格內容
table_content = []
for idx, row in results_df.iterrows():
    table_content.append([
        row['Number of Features'],
        row['Selected Features'],
        f"{row['RMSE']:.6f}",
        f"{row['R-squared']:.6f}"
    ])

columns = ['Number of Features', 'Selected Features', 'RMSE', 'R-squared']

table = ax_tbl.table(
    cellText=table_content,
    colLabels=columns,
    cellLoc='left',
    loc='center',
    colWidths=[0.18, 0.54, 0.14, 0.14]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# 美化表格樣式（經典簡潔排版）
for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_linewidth(0)
    cell.set_facecolor('white')
    
    # 數字欄位靠右對齊，其餘靠左
    if col_idx in [2, 3]:
        cell.set_text_props(ha='right')
    else:
        cell.set_text_props(ha='left')
        
    if row_idx == 0:
        cell.set_text_props(weight='bold', color='#111111')
        cell.visible_edges = 'TB'  # 標題上下框線
        cell.set_edgecolor('#555555')
        cell.set_linewidth(1.0)
    elif row_idx == len(results_df):
        cell.visible_edges = 'B'   # 底端結尾框線
        cell.set_edgecolor('#555555')
        cell.set_linewidth(1.0)
    else:
        cell.visible_edges = ''

plt.tight_layout()
plt.savefig('features_table.png', bbox_inches='tight', facecolor='white')
print("2. 表格圖已儲存為 features_table.png")

plt.show()
