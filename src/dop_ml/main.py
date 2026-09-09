import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from data_loader import load_data
from feature_extraction import extract_tsfresh_features


def train_and_evaluate_models(X, y):
    """
    Обучает 4 модели:
      - RandomForest
      - GradientBoosting
      - XGBoost
      - CatBoost
    Возвращает словарь с MAE train/val.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=1),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=1
        ),
        "CatBoost": CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            iterations=1000,
            random_state=1,
            verbose=0
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        model.fit(X_train_sc, y_train)

        y_pred_train = model.predict(X_train_sc)
        y_pred_val = model.predict(X_val_sc)

        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_val = mean_absolute_error(y_val, y_pred_val)

        results[name] = (mae_train, mae_val)

        print(f"MAE train = {mae_train:.2f}")
        print(f"MAE val   = {mae_val:.2f}")

    return results


def main():
    print("Загрузка данных...")
    X_raw, y_data = load_data()

    print("Извлечение признаков TSFresh (MinimalFCParameters)...")
    X_feat = extract_tsfresh_features(X_raw, y_data)

    print("Обучение моделей и сравнение...")
    results = train_and_evaluate_models(X_feat, y_data['predicted'])

    print("\nИтоговая таблица MAE:")
    for model, (mae_train, mae_val) in results.items():
        print(f"{model:15s} | train={mae_train:.2f} | val={mae_val:.2f}")


if __name__ == '__main__':
    main()
