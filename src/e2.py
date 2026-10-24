# src/e2.py
import pandas as pd
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PATH, TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, SPLIT_INDICES_PATH

def run_e2():
    df = pd.read_csv(DATA_PATH).drop([13], errors='ignore')
    with open(SPLIT_INDICES_PATH) as f:
        idx = json.load(f)
        
    X_train = df.loc[idx["train_idx"]].drop(TARGET_COL, axis=1)
    y_train = df.loc[idx["train_idx"]][TARGET_COL]
    X_test = df.loc[idx["test_idx"]].drop(TARGET_COL, axis=1)
    y_test = df.loc[idx["test_idx"]][TARGET_COL]
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUMERIC_COLS),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_COLS)
    ])
    
    model = Pipeline([
        ('pre', preprocessor),
        ('reg', TransformedTargetRegressor(
            regressor=GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='neg_mean_squared_error'), 
            func=np.log1p, 
            inverse_func=np.expm1
        ))
    ])
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("--- E2: Ridge ---")
    print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2:  {r2_score(y_test, preds):.4f}")

if __name__ == "__main__":
    run_e2()