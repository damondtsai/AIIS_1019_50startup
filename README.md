# Kaggle 50 Startups 多元線性迴歸與特徵選擇互動分析儀表板

本專案實作 **Kaggle 50 Startups** 資料集的多元線性迴歸模型與完整特徵選擇工作流程，遵循 **CRISP-DM** 規範，整合三種經典特徵篩選法（ANOVA F-test、RFE 遞迴特徵消除、LassoCV L1 正則化），並提供 1～5 個特徵的逐步效能評估、交叉驗證推薦與深色卡片式即時利潤預測器。

- **線上成果網站 (GitHub Pages)**: `https://damondtsai.github.io/AIIS_1019_50startup/`
- **專案儲存庫**: [https://github.com/damondtsai/AIIS_1019_50startup](https://github.com/damondtsai/AIIS_1019_50startup)
- **參考成果網站**: [https://chenyichencyc.github.io/1019_50startup/](https://chenyichencyc.github.io/1019_50startup/)
- **參考儲存庫**: [https://github.com/chenyichencyc/1019_50startup](https://github.com/chenyichencyc/1019_50startup)

---

## 專案架構與檔案說明

```
AIIS_1019_50startup/
├── index.html                   # 儀表板首頁入口 (支援 GitHub Pages 靜態託管)
├── css/
│   └── style.css                # 深藍主題、青藍重點色、玻璃擬態卡片與響應式樣式
├── js/
│   └── main.js                  # 前端互動邏輯、Chart.js 圖表渲染、即時預測器計算
├── data/
│   ├── 50_Startups.csv          # 原始資料集 (50 筆新創公司支出與利潤資料)
│   └── analysis_results.json    # 由 Python 實際運算輸出的分析結果 JSON
├── scripts/
│   └── build_results.py         # 資料驗證、模型訓練、交叉驗證與特徵選擇分析管線
├── features_performance.png     # Python 產生的特徵數量效能評估圖 (RMSE & R²)
├── requirements.txt             # Python 環境相依套件
├── .gitignore                   # Git 忽略設定 (.venv, 快取, 暫存檔)
└── README.md                    # 專案詳細說明文件
```

---

## 1. 資料處理與建模方法

1. **資料集驗證**：
   - 來源包含 `R&D Spend`、`Administration`、`Marketing Spend`、`State` 與目標變數 `Profit`。
   - 經 `scripts/build_results.py` 嚴格檢驗，全表共 50 筆資料，無遺漏值、非數值或非法負數。
2. **One-Hot Encoding**：
   - 對類別變數 `State`（包含 California, Florida, New York）執行 One-Hot Encoding，設定 `drop_first=True`。
   - 以 California 作為基準對照組，產生 `State_Florida` 與 `State_New York` 兩個二元虛擬變數。原始 4 個輸入欄位經編碼後成為 **5 個模型自變數**，有效避免共線性與虛擬變數陷阱。
3. **資料切分與交叉驗證**：
   - 固定使用 `train_test_split(test_size=0.2, random_state=0)`，切分為 40 筆訓練集與 10 筆測試集。
   - 交叉驗證（5-Fold CV）僅在訓練集上執行，作為模型結構推薦的客觀標準。
4. **特徵數量逐步評估（1～5 特徵）**：
   - 依序加入：`R&D Spend` → `Marketing Spend` → `State_New York` → `State_Florida` → `Administration`。
   - 評估結果如下表：

| 特徵數量 (k) | 選用特徵組合 | 訓練集 5-Fold CV RMSE | 測試集 Test RMSE | 測試集 Test R² | 評估狀態 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `[R&D Spend]` | $10,163.46 | 8274.868018 | 0.946459 | 候選模型 |
| **2** | **`[R&D Spend, Marketing Spend]`** | **$9,906.12** | **8198.797191** | **0.947439** | **⭐ 交叉驗證推薦最佳模型** |
| 3 | `[R&D Spend, Marketing Spend, State_New York]` | $10,278.93 | 8309.059683 | 0.946015 | 候選模型 |
| 4 | `[R&D Spend, Marketing Spend, State_New York, State_Florida]` | $10,430.08 | 8409.916714 | 0.944697 | 候選模型 |
| 5 | `[R&D Spend, Marketing Spend, State_New York, State_Florida, Administration]` | $10,612.95 | 9137.990153 | 0.934707 | 全變數模型 (過擬合) |

5. **三種特徵選擇方法（均在訓練集 X_train 上獨立進行）**：
   - **ANOVA F-test (過濾法)**：計算單變量 F-Score，排名為 `R&D Spend (652.39)` > `Marketing Spend (49.08)` > `Administration (4.73)` > `State_New York (0.91)` > `State_Florida (0.01)`。
   - **RFE 遞迴特徵消除 (包裝法)**：在標準化特徵上訓練，依係數絕對值逐步消除，最終保留第一名為 `R&D Spend`，其次為 `Marketing Spend`。
   - **LassoCV (嵌入法)**：透過 5-Fold CV 尋得最佳懲罰係數 $\alpha$，L1 正則化自動將 `Administration` 與兩州別變數壓縮歸零，僅保留 `R&D Spend` 與 `Marketing Spend`。

---

## 2. 本機執行與建置步驟

### 步驟 1：建立虛擬環境並安裝依賴
```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 安裝相依套件
pip install -r requirements.txt
```

### 步驟 2：執行分析腳本產生 JSON 結果檔
```bash
python scripts/build_results.py
```
執行後會在 `data/` 目錄產生 `analysis_results.json`，並在根目錄輸出 `features_performance.png`。

### 步驟 3：啟動本機 HTTP 伺服器預覽網頁
因為前端使用 `fetch('data/analysis_results.json')` 載入運算數據，請透過 HTTP 伺服器開啟（避免瀏覽器 `file://` 跨域限制）：
```bash
python -m http.server 8080
```
在瀏覽器開啟：[http://localhost:8080](http://localhost:8080) 即可完整操作互動儀表板。

---

## 3. GitHub Pages 設定與部署流程

本專案所有資源皆使用相對路徑，完全相容 GitHub Pages 的子路徑架構（例如 `/AIIS_1019_50startup/`）。

### 啟用 GitHub Pages 步驟：
1. 進入 GitHub 儲存庫頁面：`https://github.com/damondtsai/AIIS_1019_50startup`
2. 點選上方頁籤 **「Settings」**。
3. 在左側選單點選 **「Pages」**。
4. 在 **「Build and deployment」** 區塊：
   - **Source** 選擇：`Deploy from a branch`。
   - **Branch** 選擇：`main` 分支，資料夾選擇 `/ (root)`。
5. 點擊 **「Save」** 儲存設定。
6. 等待 1～2 分鐘 GitHub Actions 部署完成後，即可透過以下網址公開瀏覽：
   👉 **`https://damondtsai.github.io/AIIS_1019_50startup/`**

---

## 4. 分析限制與注意事項

1. **小樣本敏感性**：資料集總量為 50 筆，80/20 切分後測試集僅有 10 筆。測試指標受個別極端樣本影響較大，請勿僅憑單次測試集指標斷言特徵好壞，應優先參考交叉驗證（CV）結果。
2. **前後端完全解耦**：前端 JavaScript 嚴禁手動寫死任何評估數據，所有數值均由 Python 腳本實際計算並輸出 JSON，確保資料科學流程的可重現性（Reproducibility）。
