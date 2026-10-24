# src/bootstrap_ci.py
import pandas as pd
import numpy as np
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PATH, TARGET_COL, CATEGORICAL_COLS, SPLIT_INDICES_PATH

def get_preprocessor(numeric_cols):
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([
        ('num', numeric, numeric_cols),
        ('cat', categorical, CATEGORICAL_COLS)
    ])

def run_bootstrap():
    print("Загрузка данных...")
    df = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
        
    X_train = df.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    y_train = df.loc[idx["train_idx"]][TARGET_COL]
    X_test = df.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    y_test = df.loc[idx["test_idx"]][TARGET_COL]
    
    # Лучшая модель из E4 (замена признаков на Vproxy)
    numeric_vproxy = ["Length1", "Length2", "Vproxy"]
    preprocessor = get_preprocessor(numeric_vproxy)
    
    model = Pipeline([
        ('pre', preprocessor),
        ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    
    print("Обучение модели...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Точечные оценки
    mse_orig = mean_squared_error(y_test, preds)
    mae_orig = mean_absolute_error(y_test, preds)
    r2_orig = r2_score(y_test, preds)
    
    print(f"\nТочечные оценки на тесте:")
    print(f"MSE: {mse_orig:.2f}")
    print(f"MAE: {mae_orig:.2f}")
    print(f"R2:  {r2_orig:.4f}")
    
    # Bootstrap
    n_iterations = 1000
    n_size = len(y_test)
    
    mse_scores = []
    mae_scores = []
    r2_scores = []
    
    print(f"\nЗапуск Bootstrap ({n_iterations} итераций)...")
    np.random.seed(42)
    for i in range(n_iterations):
        # Сэмплируем индексы с возвращением
        indices = np.random.choice(n_size, n_size, replace=True)
        y_true_sample = y_test.iloc[indices]
        y_pred_sample = preds[indices]
        
        mse_scores.append(mean_squared_error(y_true_sample, y_pred_sample))
        mae_scores.append(mean_absolute_error(y_true_sample, y_pred_sample))
        r2_scores.append(r2_score(y_true_sample, y_pred_sample))
        
    # 95% доверительные интервалы (2.5% и 97.5% перцентили)
    mse_ci = np.percentile(mse_scores, [2.5, 97.5])
    mae_ci = np.percentile(mae_scores, [2.5, 97.5])
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])
    
    print("\n--- 95% Доверительные интервалы (Bootstrap) ---")
    print(f"MSE: [{mse_ci[0]:.2f}, {mse_ci[1]:.2f}]")
    print(f"MAE: [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
    print(f"R2:  [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

if __name__ == "__main__":
    run_bootstrap()