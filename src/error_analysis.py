# src/error_analysis.py
import pandas as pd
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from config import DATA_PATH, TARGET_COL, CATEGORICAL_COLS, SPLIT_INDICES_PATH

def run_error_analysis():
    df = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
    
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
        
    X_train = df.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    y_train = df.loc[idx["train_idx"]][TARGET_COL]
    X_test = df.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    y_test = df.loc[idx["test_idx"]][TARGET_COL]
    
    numeric_vproxy = ["Length1", "Length2", "Vproxy"]
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric_vproxy),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_COLS)
    ])
    
    model = Pipeline([
        ('pre', preprocessor),
        ('reg', HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results = X_test.copy()
    results['True_Weight'] = y_test
    results['Predicted_Weight'] = preds
    results['Absolute_Error'] = np.abs(y_test - preds)
    
    # 1. ТОП-5 ХУДШИХ ПРОГНОЗОВ
    top_5_errors = results.sort_values(by='Absolute_Error', ascending=False).head(5)
    print("\n--- ТОП-5 ХУДШИХ ПРОГНОЗОВ ---")
    cols = ['Species', 'Length3', 'Height', 'Width', 'True_Weight', 'Predicted_Weight', 'Absolute_Error']
    print(top_5_errors[cols].to_string(index=False))
    
    # 2. MAE ПО ВИДАМ
    print("\n--- MAE ПО ВИДАМ ---")
    mae_by_species = results.groupby('Species')['Absolute_Error'].mean().sort_values(ascending=False)
    print(mae_by_species.to_string())
    
    # 3. MAE ПО ДИАПАЗОНАМ РАЗМЕРА
    print("\n--- MAE ПО ДИАПАЗОНАМ РАЗМЕРА ---")
    def size_category(weight):
        if weight < 300:
            return "Мелкие (<300 г)"
        elif weight <= 800:
            return "Средние (300-800 г)"
        else:
            return "Крупные (>800 г)"
    
    results['Size_Category'] = results['True_Weight'].apply(size_category)
    mae_by_size = results.groupby('Size_Category')['Absolute_Error'].mean()
    print(mae_by_size.to_string())
    
    # 4. ПРОВЕРКА НА ОТРИЦАТЕЛЬНЫЕ ПРЕДСКАЗАНИЯ
    print("\n--- ПРОВЕРКА НА ОТРИЦАТЕЛЬНЫЕ ПРЕДСКАЗАНИЯ ---")
    negative_preds = (results['Predicted_Weight'] < 0).sum()
    print(f"Количество отрицательных предсказаний: {negative_preds}")

if __name__ == "__main__":
    run_error_analysis()