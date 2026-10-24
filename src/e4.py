# src/e4.py
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, SPLIT_INDICES_PATH

def save_residuals_plot(y_true, y_pred, filename):
    plt.figure(figsize=(8, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Weight")
    plt.ylabel("Residuals")
    plt.title(f"Residuals: {filename}")
    # Создаем папку для графиков, если ее нет
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    plt.savefig(os.path.join("reports", "figures", filename))
    plt.close()

def run_e4():
    df = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    
    # Создаем производный признак Vproxy
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    
    # 1. Проверка корреляций (требование из "Внимание" на скрине)
    print("\n--- Корреляция Vproxy с исходными признаками ---")
    corr_matrix = df[["Vproxy", "Length3", "Height", "Width"]].corr()
    print(corr_matrix["Vproxy"])
    
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
        
    X_train = df.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    y_train = df.loc[idx["train_idx"]][TARGET_COL]
    X_test = df.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    y_test = df.loc[idx["test_idx"]][TARGET_COL]
    
    # Общий препроцессор для категорий
    cat_pipeline = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    
    # === ВАРИАНТ А: Добавление Vproxy к исходным признакам ===
    numeric_add = ["Length1", "Length2", "Length3", "Height", "Width", "Vproxy"]
    preprocessor_add = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric_add),
        ('cat', cat_pipeline, CATEGORICAL_COLS)
    ])
    
    model_add = Pipeline([
        ('pre', preprocessor_add),
        ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    model_add.fit(X_train, y_train)
    preds_add = model_add.predict(X_test)
    
    print("\n--- E4 (Вариант А): Бустинг + Vproxy (добавлен) ---")
    print(f"MSE: {mean_squared_error(y_test, preds_add):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, preds_add):.2f}")
    print(f"R2:  {r2_score(y_test, preds_add):.4f}")
    save_residuals_plot(y_test, preds_add, "e4_add_residuals.png")

    # === ВАРИАНТ Б: Замена группы признаков на Vproxy ===
    # Оставляем только Length1, Length2 и Vproxy (убираем Length3, Height, Width)
    numeric_replace = ["Length1", "Length2", "Vproxy"]
    preprocessor_replace = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric_replace),
        ('cat', cat_pipeline, CATEGORICAL_COLS)
    ])
    
    model_replace = Pipeline([
        ('pre', preprocessor_replace),
        ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    model_replace.fit(X_train, y_train)
    preds_replace = model_replace.predict(X_test)
    
    print("\n--- E4 (Вариант Б): Бустинг (Vproxy заменил L3, H, W) ---")
    print(f"MSE: {mean_squared_error(y_test, preds_replace):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, preds_replace):.2f}")
    print(f"R2:  {r2_score(y_test, preds_replace):.4f}")
    save_residuals_plot(y_test, preds_replace, "e4_replace_residuals.png")

if __name__ == "__main__":
    run_e4()