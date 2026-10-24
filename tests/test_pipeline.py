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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from config import DATA_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, SPLIT_INDICES_PATH

def test_split_indices_do_not_overlap():
    """Тест 1: Индексы train и test не пересекаются и покрывают весь датасет."""
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
    
    train_idx = set(idx["train_idx"])
    test_idx = set(idx["test_idx"])
    
    # Проверяем, что пересечения нет
    assert train_idx.isdisjoint(test_idx), "Train и Test индексы пересекаются!"
    
    # Проверяем, что вместе они дают весь датасет (минус выброс)
    df = pd.read_csv(DATA_PATH)
    df = df.drop([13], errors='ignore')
    assert len(train_idx | test_idx) == len(df), "Индексы не покрывают весь датасет"

def test_pipeline_handles_unknown_category():
    """Тест 2: Пайплайн не падает на неизвестной категории."""
    # Создаем простой пайплайн
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric, NUMERIC_COLS),
        ('cat', categorical, CATEGORICAL_COLS)
    ])
    
    model = Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
    
    # Данные для тренировки (известные виды)
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
    
    # Новые данные с НЕИЗВЕСТНЫМ видом рыбы (например, "Shark")
    X_test = pd.DataFrame({
        'Species': ['Shark'],  # <-- Неизвестная категория
        'Length1': [25.0],
        'Length2': [27.0],
        'Length3': [30.0],
        'Height': [10.0],
        'Width': [4.0]
    })
    
    # Если пайплайн не упал, тест пройден
    predictions = model.predict(X_test)
    assert len(predictions) == 1
    assert np.isfinite(predictions[0]), "Предсказание не является конечным числом"