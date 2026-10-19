#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/build_results.py
Kaggle 50 Startups 多元線性迴歸與特徵選擇完整分析管線
由資料實際運算產生 JSON 供前端使用，不依賴任何硬編碼數值。
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import f_regression, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DATA_CSV_PATH = os.path.join("data", "50_Startups.csv")
REMOTE_CSV_URL = "https://raw.githubusercontent.com/chenyichencyc/1019_50startup/main/50_Startups.csv"
OUTPUT_JSON_PATH = os.path.join("data", "analysis_results.json")

def load_and_validate_data():
    """載入並嚴格驗證 CSV 欄位與品質"""
    if not os.path.exists(DATA_CSV_PATH):
        print(f"[*] 本機未找到 {DATA_CSV_PATH}，正在從遠端下載：{REMOTE_CSV_URL}...")
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(REMOTE_CSV_URL, DATA_CSV_PATH)
        print("[+] 下載完成。")

    df = pd.read_csv(DATA_CSV_PATH)
    df.columns = df.columns.str.strip()

    expected_cols = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[錯誤] CSV 資料集缺少預期欄位: {missing_cols}")

    if len(df) != 50:
        raise ValueError(f"[錯誤] CSV 資料筆數為 {len(df)}，與預期 50 筆不符！")

    # 檢查遺漏值
    null_counts = df[expected_cols].isnull().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        raise ValueError(f"[錯誤] 資料包含遺漏值: {null_counts}")

    # 檢查數值型欄位非數值或非法負數
    num_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"[錯誤] 數值欄位 {col} 型態異常: {df[col].dtype}")
        if (df[col] < 0).any():
            raise ValueError(f"[錯誤] 數值欄位 {col} 包含不合法負數！")

    valid_states = {"New York", "California", "Florida"}
    actual_states = set(df["State"].unique())
    if not actual_states.issubset(valid_states):
        raise ValueError(f"[錯誤] 發現未知州別: {actual_states - valid_states}")

    print(f"[+] 資料驗證成功！共 {len(df)} 筆，無遺漏值，數值與類別皆合法。")
    return df

def run_analysis():
    df = load_and_validate_data()

    # 資料統計摘要
    stats_summary = {
        "total_samples": int(len(df)),
        "states_distribution": df["State"].value_counts().to_dict(),
        "profit": {
            "mean": float(df["Profit"].mean()),
            "std": float(df["Profit"].std()),
            "min": float(df["Profit"].min()),
            "max": float(df["Profit"].max())
        },
        "rd_spend": {
            "mean": float(df["R&D Spend"].mean()),
            "std": float(df["R&D Spend"].std()),
            "min": float(df["R&D Spend"].min()),
            "max": float(df["R&D Spend"].max())
        },
        "admin_spend": {
            "mean": float(df["Administration"].mean()),
            "std": float(df["Administration"].std()),
            "min": float(df["Administration"].min()),
            "max": float(df["Administration"].max())
        },
        "marketing_spend": {
            "mean": float(df["Marketing Spend"].mean()),
            "std": float(df["Marketing Spend"].std()),
            "min": float(df["Marketing Spend"].min()),
            "max": float(df["Marketing Spend"].max())
        }
    }

    # 特徵分離與 One-Hot Encoding (drop_first=True)
    # California 為被剔除的基準組 (Reference Category)
    X = df.drop(columns=["Profit"])
    y = df["Profit"]

    X_encoded = pd.get_dummies(X, columns=["State"], drop_first=True, dtype=float)
    encoded_feature_names = list(X_encoded.columns)
    # 預期特徵: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']

    # 固定資料切分: test_size=0.2, random_state=0
    # 預期訓練集 40 筆，測試集 10 筆
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=0
    )

    if len(X_train) != 40 or len(X_test) != 10:
        raise ValueError(f"[錯誤] 切分筆數異常: train={len(X_train)}, test={len(X_test)}")

    # 5-fold 交叉驗證設定 (僅在訓練集進行)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # ==============================================================
    # 1. 逐步特徵評估 (1 ~ 5 個特徵，依參考專案順序)
    # R&D Spend -> Marketing Spend -> State_New York -> State_Florida -> Administration
    # ==============================================================
    feature_order = [
        "R&D Spend",
        "Marketing Spend",
        "State_New York",
        "State_Florida",
        "Administration"
    ]

    stepwise_results = []
    trained_models = {}

    for k in range(1, len(feature_order) + 1):
        selected_feats = feature_order[:k]

        # 訓練集 5-fold 交叉驗證 RMSE
        cv_scores = -cross_val_score(
            LinearRegression(),
            X_train[selected_feats],
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error"
        )
        cv_rmse = float(np.sqrt(np.mean(cv_scores)))

        # 於訓練集擬合模型
        model = LinearRegression()
        model.fit(X_train[selected_feats], y_train)

        # 於測試集進行最終評估
        y_pred = model.predict(X_test[selected_feats])
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))

        coef_dict = {f: float(c) for f, c in zip(selected_feats, model.coef_)}
        model_info = {
            "k": k,
            "selected_features": selected_feats,
            "cv_rmse": round(cv_rmse, 4),
            "test_rmse": round(test_rmse, 6),
            "test_r2": round(test_r2, 6),
            "intercept": float(model.intercept_),
            "coefficients": coef_dict
        }
        stepwise_results.append(model_info)
        trained_models[f"k_{k}"] = model_info

    # 依訓練集 CV RMSE 選擇推薦模型
    best_cv_model = min(stepwise_results, key=lambda m: m["cv_rmse"])
    # 另外標出此測試切分中 Test RMSE 最低的組合
    best_test_model = min(stepwise_results, key=lambda m: m["test_rmse"])

    # ==============================================================
    # 2. 三種特徵選擇方法 (ANOVA F-test, RFE, LassoCV)
    # 全部只在訓練集 X_train 上進行！
    # ==============================================================

    # 方法 1：ANOVA F-test (Filter Method)
    f_scores, p_values = f_regression(X_train, y_train)
    anova_records = []
    for f_name, f_val, p_val in zip(encoded_feature_names, f_scores, p_values):
        anova_records.append({
            "feature": f_name,
            "f_score": round(float(f_val), 4),
            "p_value": float(p_val)
        })
    anova_records.sort(key=lambda x: x["f_score"], reverse=True)

    # 方法 2：RFE 遞迴特徵消除 (Wrapper Method)
    # 使用標準化消除量綱尺度影響
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)

    rfe_records = []
    for f_name, rank in zip(encoded_feature_names, rfe.ranking_):
        rfe_records.append({
            "feature": f_name,
            "ranking": int(rank),
            "priority_score": int(len(encoded_feature_names) + 1 - rank)
        })
    rfe_records.sort(key=lambda x: x["ranking"])

    # 方法 3：LassoCV (Embedded Method)
    lasso_cv = LassoCV(cv=5, random_state=0, max_iter=50000).fit(X_train_scaled, y_train)
    lasso_records = []
    for f_name, coef in zip(encoded_feature_names, lasso_cv.coef_):
        lasso_records.append({
            "feature": f_name,
            "coef": round(float(coef), 4),
            "abs_coef": round(float(abs(coef)), 4),
            "selected": bool(abs(coef) > 1e-4)
        })
    lasso_records.sort(key=lambda x: x["abs_coef"], reverse=True)

    # ==============================================================
    # 3. 預測器所需的模型參數包 (包含 CV 建議模型、雙特徵示範、五特徵完整模型)
    # ==============================================================
    predictors = {
        "cv_recommended": {
            "name": f"交叉驗證推薦模型 (k={best_cv_model['k']})",
            "k": best_cv_model["k"],
            "features": best_cv_model["selected_features"],
            "intercept": best_cv_model["intercept"],
            "coefficients": best_cv_model["coefficients"],
            "cv_rmse": best_cv_model["cv_rmse"],
            "test_rmse": best_cv_model["test_rmse"],
            "test_r2": best_cv_model["test_r2"]
        },
        "two_features": {
            "name": "雙特徵示範模型 [R&D, Marketing]",
            "k": 2,
            "features": trained_models["k_2"]["selected_features"],
            "intercept": trained_models["k_2"]["intercept"],
            "coefficients": trained_models["k_2"]["coefficients"],
            "cv_rmse": trained_models["k_2"]["cv_rmse"],
            "test_rmse": trained_models["k_2"]["test_rmse"],
            "test_r2": trained_models["k_2"]["test_r2"]
        },
        "full_five": {
            "name": "完整五特徵模型 (All Features)",
            "k": 5,
            "features": trained_models["k_5"]["selected_features"],
            "intercept": trained_models["k_5"]["intercept"],
            "coefficients": trained_models["k_5"]["coefficients"],
            "cv_rmse": trained_models["k_5"]["cv_rmse"],
            "test_rmse": trained_models["k_5"]["test_rmse"],
            "test_r2": trained_models["k_5"]["test_r2"]
        }
    }

    # 輸出資料結構
    output_data = {
        "metadata": {
            "dataset_name": "Kaggle 50 Startups",
            "total_records": 50,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "original_columns": ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"],
            "encoded_features": encoded_feature_names,
            "reference_category": "California (drop_first=True)",
            "split_random_state": 0,
            "test_ratio": 0.2
        },
        "summary": stats_summary,
        "stepwise_results": stepwise_results,
        "best_cv_model_k": best_cv_model["k"],
        "best_test_model_k": best_test_model["k"],
        "feature_selection_methods": {
            "anova": {
                "name": "ANOVA F-test (過濾法 Filter)",
                "description": "計算各特徵與 Profit 之間的單變量線性相關 F 分數與顯著性 p-value。分數越高代表單獨對利潤的線性解釋力越強。",
                "results": anova_records
            },
            "rfe": {
                "name": "RFE 遞迴特徵消除 (包裝法 Wrapper)",
                "description": "在訓練集上擬合標準化線性迴歸，反覆評估係數絕對值並剔除權重最小特徵，排名 1 為最後保留的最核心特徵。",
                "results": rfe_records
            },
            "lasso": {
                "name": "LassoCV L1 正則化 (嵌入法 Embedded)",
                "best_alpha": round(float(lasso_cv.alpha_), 4),
                "description": "透過 5-fold 交叉驗證挑選最佳懲罰係數 alpha，藉由 L1 稀疏約束自動將無效特徵係數直接壓縮歸零。",
                "results": lasso_records
            }
        },
        "predictors": predictors,
        "sample_records": df.head(10).to_dict(orient="records")
    }

    # 寫入 JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[+] 成功輸出分析結果至 {OUTPUT_JSON_PATH}")
    print(f"[*] 交叉驗證選定最佳模型: k={best_cv_model['k']} (CV RMSE: {best_cv_model['cv_rmse']})")
    print(f"[*] 測試集切分最低 RMSE: k={best_test_model['k']} (Test RMSE: {best_test_model['test_rmse']})")

if __name__ == "__main__":
    run_analysis()
