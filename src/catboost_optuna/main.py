import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import optuna

from data_loader import load_data
from feature_extraction import extract_tsfresh_features
from objective import objective


def main():
    print("Загрузка данных...")
    X_raw, y_data = load_data()

    print("Извлечение признаков TSFresh...")
    X_feat = extract_tsfresh_features(X_raw, y_data)

    X = X_feat.values
    y = y_data["predicted"].values
    groups = y_data.index.values

    print("\nЗапуск Optuna с GroupKFold...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=15,
        n_jobs=-1,
    )

    print("\nЛучшие параметры CatBoost (Optuna + GroupKFold):")
    print(study.best_params)
    print("Лучший средний MAE (val):", round(study.best_value, 2))

    best_trial = study.best_trial
    print("Средний MAE train:", round(best_trial.user_attrs["mae_train_mean"], 2))
    print("Средний MAE val:", round(best_trial.user_attrs["mae_val_mean"], 2))

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    best_model = CatBoostRegressor(
        **study.best_params,
        loss_function="MAE",
        random_state=1,
        verbose=0,
    )

    best_model.fit(X_sc, y)

    print("\nФинальная модель обучена на всех данных.")


if __name__ == "__main__":
    main()
