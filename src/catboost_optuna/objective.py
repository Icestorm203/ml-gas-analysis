import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor


def objective(trial, X, y, groups):
    """
    Целевая функция для Optuna:
    - подбирает гиперпараметры CatBoostRegressor,
    - оценивает качество по GroupKFold,
    - возвращает средний MAE по валидации.
    Дополнительно сохраняет средние MAE по train/val в trial.user_attrs.
    """
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "iterations": trial.suggest_int("iterations", 300, 800),
        "loss_function": "MAE",
        "random_state": 1,
        "verbose": 0,
        "od_type": "Iter",
        "od_wait": 50,
        "use_best_model": True,
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 10
        ),
    }

    cv = GroupKFold(n_splits=5)
    maes_val = []
    maes_train = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train_sc,
            y_train,
            eval_set=(X_val_sc, y_val),
            early_stopping_rounds=50,
        )

        y_pred_val = model.predict(X_val_sc)
        y_pred_train = model.predict(X_train_sc)

        maes_val.append(mean_absolute_error(y_val, y_pred_val))
        maes_train.append(mean_absolute_error(y_train, y_pred_train))

    trial.set_user_attr("mae_train_mean", np.mean(maes_train))
    trial.set_user_attr("mae_val_mean", np.mean(maes_val))

    return np.mean(maes_val)
