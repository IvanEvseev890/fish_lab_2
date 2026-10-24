# src/evaluate.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import TARGET_COL

def evaluate_model():
    print("Загрузка модели...")
    model = joblib.load("models/model.joblib")
    
    print("Загрузка данных для оценки...")
    # Используем демо-данные
    df = pd.read_csv("fish_holdout_demo.csv")
    
    # Добавляем Vproxy (так как модель обучена на нём)
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    
    X = df.drop(TARGET_COL, axis=1)
    y_true = df[TARGET_COL]
    
    print("Предсказание...")
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nРезультаты на демо-выборке:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2:  {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()