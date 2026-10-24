# src/run_mlflow.py
import pandas as pd
import numpy as np
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, SPLIT_INDICES_PATH

# Настройка MLflow (пишем напрямую в SQLite базу)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Fish_Weight_Lab2")

def load_data():
    df = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
    X_train = df.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    y_train = df.loc[idx["train_idx"]][TARGET_COL]
    X_test = df.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    y_test = df.loc[idx["test_idx"]][TARGET_COL]
    return X_train, X_test, y_train, y_test

def get_preprocessor(numeric_cols):
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([
        ('num', numeric, numeric_cols),
        ('cat', categorical, CATEGORICAL_COLS)
    ])

def save_residuals_plot(y_true, y_pred, filename):
    plt.figure(figsize=(8, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Weight")
    plt.ylabel("Residuals")
    plt.title(f"Residuals: {filename}")
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    path = os.path.join("reports", "figures", filename)
    plt.savefig(path)
    plt.close()
    return path

def evaluate_and_log(run_name, model, X_train, y_train, X_test, y_test, params=None, save_model=False):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        if params:
            mlflow.log_params(params)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse_mean = -cv_scores.mean()
        cv_mse_std = cv_scores.std()
        
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_mse_mean", cv_mse_mean)
        mlflow.log_metric("cv_mse_std", cv_mse_std)
        
        if save_model:
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format="pickle"
                )
            except Exception as e:
                print(f"  (Модель не сохранена из-за ограничений: {e})")
        
        plot_path = save_residuals_plot(y_test, preds, f"{run_name}_residuals.png")
        mlflow.log_artifact(plot_path)
        
        print(f"[{run_name}] MSE: {mse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f} | CV MSE: {cv_mse_mean:.2f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = get_preprocessor(NUMERIC_COLS)
    
    # E0
    e0 = Pipeline([('pre', preprocessor), ('reg', DummyRegressor(strategy="mean"))])
    evaluate_and_log("E0_Dummy", e0, X_train, y_train, X_test, y_test, save_model=True)
    
    # E1
    e1 = Pipeline([
        ('pre', preprocessor),
        ('reg', TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1))
    ])
    evaluate_and_log("E1_Linear_LogTarget", e1, X_train, y_train, X_test, y_test, save_model=True)
    
    # E2 (без сохранения модели из-за GridSearchCV)
    e2 = Pipeline([
        ('pre', preprocessor),
        ('reg', TransformedTargetRegressor(
            regressor=GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='neg_mean_squared_error'),
            func=np.log1p, inverse_func=np.expm1
        ))
    ])
    evaluate_and_log("E2_Ridge", e2, X_train, y_train, X_test, y_test, 
                     params={"model": "Ridge", "alpha_grid": "[0.01, 0.1, 1.0, 10.0, 100.0]"}, 
                     save_model=False)
    
    # E3
    e3 = Pipeline([('pre', preprocessor), ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))])
    evaluate_and_log("E3_HistGradientBoosting", e3, X_train, y_train, X_test, y_test, 
                     params={"max_iter": 100, "max_depth": 5, "learning_rate": 0.1}, save_model=True)
    
    # E4 (замена на Vproxy)
    df_full = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    df_full["Vproxy"] = df_full["Length3"] * df_full["Height"] * df_full["Width"]
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
    X_train_v = df_full.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    X_test_v = df_full.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    
    numeric_vproxy = ["Length1", "Length2", "Vproxy"]
    preprocessor_vproxy = get_preprocessor(numeric_vproxy)
    
    e4 = Pipeline([('pre', preprocessor_vproxy), ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))])
    evaluate_and_log("E4_HGB_Vproxy_Replace", e4, X_train_v, y_train, X_test_v, y_test, 
                     params={"max_iter": 100, "max_depth": 5, "learning_rate": 0.1, "features": "Length1, Length2, Vproxy"}, 
                     save_model=True)

    # Сохранение лучшей модели для evaluate.py
    os.makedirs("models", exist_ok=True)
    joblib.dump(e4, "models/model.joblib")
    print("Модель сохранена в models/model.joblib")

    print("\nВсе эксперименты завершены. Запустите 'mlflow server --backend-store-uri sqlite:///mlflow.db' для просмотра.")