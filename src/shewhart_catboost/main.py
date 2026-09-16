import numpy as np
import pandas as pd
import shap
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from tsfresh.feature_extraction import extract_features, MinimalFCParameters

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH = os.path.join(BASE_DIR, 'data') + '\\'

def load_data():
    y_data = pd.read_csv(PATH + 'train.csv', index_col='id')

    X_data = {}
    for file_name in y_data.index:
        path = PATH + f'data_train/{file_name}'
        X_data[file_name] = pd.read_csv(path)

    X = pd.concat([X_data[file].assign(id=file) for file in y_data.index],
                  axis=0, ignore_index=True)

    return X, y_data


def extract_tsfresh_features(X, y_data):
    settings = MinimalFCParameters()

    X_feat = extract_features(
        X,
        column_id="id",
        default_fc_parameters=settings,
        n_jobs=0
    )

    X_feat = X_feat.loc[y_data.index]
    return X_feat


def tune_catboost_fast(X_train_sc, y_train):
    model = CatBoostRegressor(
        random_state=1,
        verbose=0,
        loss_function='MAE'
    )

    param_dist = {
        'depth': [4, 6],
        'learning_rate': [0.03, 0.05],
        'l2_leaf_reg': [3, 5],
        'iterations': [500, 800]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=5,
        scoring='neg_mean_absolute_error',
        cv=2,
        random_state=1,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X_train_sc, y_train)

    print("\nЛучшие параметры CatBoost:")
    print(search.best_params_)

    print("\nЛучший MAE (cv):", -search.best_score_)

    return search.best_estimator_


def compute_ucl(series, k=3):
    mean = np.mean(series)
    std = np.std(series)
    return mean + k * std


def detect_anomaly_shuhart(series, k=3):
    ucl = compute_ucl(series, k)
    return np.any(series > ucl), ucl


def hybrid_diagnosis(gas_series_dict, X_features_row, model, scaler):

    anomaly_flags = {}
    ucl_values = {}
    for gas_name, series in gas_series_dict.items():
        anomaly, ucl = detect_anomaly_shuhart(series)
        anomaly_flags[gas_name] = anomaly
        ucl_values[gas_name] = ucl

    X_scaled = scaler.transform(X_features_row)
    rul_pred = model.predict(X_scaled)[0]

    if any(anomaly_flags.values()):
        status = "Аномалия по статистическому контролю (Шухарт)"
    elif rul_pred < 50:
        status = "Критическое снижение RUL"
    elif rul_pred < 100:
        status = "Умеренная деградация"
    else:
        status = "Нормальный режим"

    return {
        "anomaly_flags": anomaly_flags,
        "ucl_values": ucl_values,
        "rul_prediction": rul_pred,
        "status": status
    }


def main():
    print("Загрузка данных...")
    X_raw, y_data = load_data()

    print("Извлечение признаков TSFresh (MinimalFCParameters)...")
    X_feat = extract_tsfresh_features(X_raw, y_data)

    print("Разделение на train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y_data['predicted'], test_size=0.25, random_state=1
    )

    print("Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    print("\nЗапуск ускоренного RandomizedSearchCV для CatBoost...")
    best_cat = tune_catboost_fast(X_train_sc, y_train)

    print("\nОценка лучшей модели...")
    y_pred_train = best_cat.predict(X_train_sc)
    y_pred_val = best_cat.predict(X_val_sc)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)

    print("\nCatBoost + Fast RandomizedSearchCV результаты:")
    print("MAE train =", round(mae_train, 2))
    print("MAE val   =", round(mae_val, 2))

    test_id = "2_trans_497.csv"
    print(f"\nГибридная диагностика для объекта {test_id}")

    gas_data = {}
    path = PATH + f"data_train/{test_id}"
    df = pd.read_csv(path)
    for gas_name in ["H2", "CO", "C2H4", "C2H2"]:
        gas_data[gas_name] = df[gas_name].values

    X_row = X_feat.loc[[test_id]]

    result = hybrid_diagnosis(gas_data, X_row, best_cat, scaler)

    print("\n--- Результаты гибридной диагностики ---")
    print("Флаги аномалий:", result["anomaly_flags"])
    print("UCL:", result["ucl_values"])
    print("Прогноз RUL:", result["rul_prediction"])
    print("Статус:", result["status"])


if __name__ == '__main__':
    main()
