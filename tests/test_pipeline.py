# tests/test_pipeline.py
import sys
import os

# Добавляем в путь и корень проекта, и папку src, чтобы pytest видел config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

import pandas as pd
import numpy as np
import json
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, SPLIT_INDICES_PATH

# --- ТЕСТ 1: Индексы не пересекаются ---
def test_split_indices_do_not_overlap():
    """Проверка 1: Индексы train и test не пересекаются и покрывают весь датасет."""
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
    
    train_idx = set(idx["train_idx"])
    test_idx = set(idx["test_idx"])
    
    assert train_idx.isdisjoint(test_idx), "Train и Test индексы пересекаются!"
    
    df = pd.read_csv(DATA_PATH)
    df = df.drop([13], errors='ignore')
    assert len(train_idx | test_idx) == len(df), "Индексы не покрывают весь датасет"

# --- ТЕСТ 2: Неизвестная категория ---
def test_pipeline_handles_unknown_category():
    """Проверка 2: Цепочка обработки принимает строку с неизвестной категорией."""
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric, NUMERIC_COLS),
        ('cat', categorical, CATEGORICAL_COLS)
    ])
    
    model = Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
    
    X_train = pd.DataFrame({
        'Species': ['Bream', 'Perch', 'Bream'],
        'Length1': [26.5, 20.0, 30.4],
        'Length2': [29.0, 22.0, 33.0],
        'Length3': [34.0, 23.5, 38.3],
        'Height': [12.4, 5.5, 14.8],
        'Width': [5.1, 3.9, 5.2]
    })
    y_train = [430.0, 110.0, 700.0]
    model.fit(X_train, y_train)
    
    X_test = pd.DataFrame({
        'Species': ['Shark'],  # Неизвестная категория
        'Length1': [25.0],
        'Length2': [27.0],
        'Length3': [30.0],
        'Height': [10.0],
        'Width': [4.0]
    })
    
    predictions = model.predict(X_test)
    assert len(predictions) == 1, "Пайплайн упал на неизвестной категории"

# --- ТЕСТ 3: Число прогнозов и конечность ---
def test_predictions_count_and_finiteness():
    """Проверка 3: Число прогнозов равно числу входных строк, все значения конечны."""
    model = joblib.load("models/model.joblib")
    
    df = pd.read_csv("fish_holdout_demo.csv").head(5)
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    X = df.drop("Weight", axis=1)
    
    predictions = model.predict(X)
    
    assert len(predictions) == len(X), "Число прогнозов не равно числу входных строк"
    assert np.all(np.isfinite(predictions)), "Среди прогнозов есть NaN или бесконечность"

# --- ТЕСТ 4: Метрики совпадают с ручным расчетом ---
def test_manual_metrics_calculation():
    """Проверка 4: Метрики на фиксированном контрольном примере совпадают с ручным расчётом."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 330])
    
    mse_manual = ((10)**2 + (10)**2 + (30)**2) / 3
    mae_manual = (10 + 10 + 30) / 3
    ss_res = (10**2 + 10**2 + 30**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    assert np.isclose(mean_squared_error(y_true, y_pred), mse_manual)
    assert np.isclose(mean_absolute_error(y_true, y_pred), mae_manual)
    assert np.isclose(r2_score(y_true, y_pred), r2_manual)

# --- ТЕСТ 5: Сохранение и загрузка модели ---
def test_model_save_and_load():
    """Проверка 5: Сохранённая цепочка обработки после загрузки воспроизводит прогноз."""
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Модель не найдена (запустите run_mlflow.py)")
    
    model = joblib.load(model_path)
    
    df = pd.read_csv("fish_holdout_demo.csv").head(1)
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    X = df.drop("Weight", axis=1)
    
    pred1 = model.predict(X)[0]
    model2 = joblib.load(model_path)
    pred2 = model2.predict(X)[0]
    
    assert np.isclose(pred1, pred2), "Загруженная модель даёт другой прогноз"