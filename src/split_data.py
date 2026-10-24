# src/split_data.py
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE, SPLIT_INDICES_PATH

def split_and_save():
    # Читаем данные
    df = pd.read_csv(DATA_PATH)
    
    # В исходном репо удаляли строку 13 как выброс (в ноутбуке это видно)
    # Сделаем это один раз и навсегда, чтобы данные были одинаковыми для всех
    df = df.drop([13], errors='ignore')
    
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    # Пункт 1: Замораживаем тестовую выборку
    # Пункт 5: Сохраняем индексы (они будут одинаковыми при каждом запуске благодаря random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Сохраняем индексы в JSON
    indices = {
        "train_idx": X_train.index.tolist(),
        "test_idx": X_test.index.tolist()
    }
    
    with open(SPLIT_INDICES_PATH, 'w') as f:
        json.dump(indices, f)
        
    print(f"Разбиение выполнено. Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Индексы сохранены в {SPLIT_INDICES_PATH}")

if __name__ == "__main__":
    split_and_save()