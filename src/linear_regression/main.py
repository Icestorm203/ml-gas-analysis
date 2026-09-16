import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score
from catboost import CatBoostRegressor
from tsfresh.feature_extraction import extract_features, MinimalFCParameters

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH = os.path.join(BASE_DIR, 'data') + '\\'


def main():
    # --- Загрузка данных ---
    y_data = pd.read_csv(PATH + 'train.csv', index_col='id')
    print(y_data.head())

    X_data = {}
    for row in y_data.iterrows():
        file_name = row[0]
        path = PATH + f'data_train/{file_name}'
        X_data[file_name] = pd.read_csv(path)
    print(X_data[file_name].head(2))

    # --- Бинаризация целевой переменной для поиска выбросов ---
    y = (y_data == 1093).astype(int)

    # --- Объединение временных рядов ---
    X = pd.concat(
        [X_data[file].assign(id=file) for file in y_data.index],
        axis=0,
        ignore_index=True
    )

    # --- Извлечение признаков TSFresh ---
    settings = MinimalFCParameters()
    X = extract_features(
        X,
        column_id="id",
        default_fc_parameters=settings
    ).loc[y.index]

    # --- Разделение на train/val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    # --- Масштабирование ---
    StSc = StandardScaler()
    X_train_sc = StSc.fit_transform(X_train)
    X_val_sc = StSc.transform(X_val)

    # --- Логистическая регрессия для поиска выбросов ---
    LogReg = LogisticRegression()
    LogReg.fit(X_train_sc, y_train)

    y_train_pred_outliers = pd.DataFrame(
        LogReg.predict(X_train_sc),
        index=y_train.index,
        columns=y_train.columns
    )
    y_val_pred_outliers = pd.DataFrame(
        LogReg.predict(X_val_sc),
        index=y_val.index,
        columns=y_val.columns
    )

    f1_train = round(f1_score(y_train, y_train_pred_outliers), 2)
    f1_val = round(f1_score(y_val, y_val_pred_outliers), 2)
    print(f'F1 on train set = {f1_train}')
    print(f'F1 on test set = {f1_val}')

    # --- Вторая часть: регрессия CatBoost ---
    y = y_data.copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    StSc = StandardScaler()
    X_train_sc = StSc.fit_transform(X_train)
    X_val_sc = StSc.transform(X_val)

    cbr = CatBoostRegressor(random_state=1, verbose=0)
    cbr.fit(X_train_sc, y_train)

    y_train_pred = pd.DataFrame(
        cbr.predict(X_train_sc),
        index=y_train.index,
        columns=y_train.columns
    )
    y_val_pred = pd.DataFrame(
        cbr.predict(X_val_sc),
        index=y_val.index,
        columns=y_val.columns
    )

    # --- Коррекция выбросов ---
    ind = y_train_pred_outliers[y_train_pred_outliers['predicted'] == 1].index
    y_train_pred.loc[ind] = 1093

    ind = y_val_pred_outliers[y_val_pred_outliers['predicted'] == 1].index
    y_val_pred.loc[ind] = 1093

    mae_train = round(mean_absolute_error(y_train, y_train_pred), 2)
    mae_val = round(mean_absolute_error(y_val, y_val_pred), 2)

    print('TSFresh + Логистическая регрессия + Градиентный бустинг')
    print(f'MAE on train set = {mae_train}')
    print(f'MAE on test set = {mae_val}')

    # --- График ---
    plt.figure(figsize=(7, 3))
    plt.hist(y_val_pred, bins=30, alpha=0.6, label='Прогнозируемые значения на тестовом наборе')
    plt.hist(y_val, bins=30, alpha=0.6, label='Истинные значения тестового набора')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
