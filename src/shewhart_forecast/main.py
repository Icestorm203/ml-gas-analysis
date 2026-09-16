import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH = os.path.join(BASE_DIR, 'data') + '\\'

GAS_COLUMNS = ['H2', 'CO', 'C2H4', 'C2H2']


def compute_control_limits(series):
    mu = series.mean()
    sigma = series.std(ddof=1)
    cl = mu
    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma
    return cl, lcl, ucl


def predict_ucl_crossing(values, ucl, n_last=30):

    if len(values) < n_last:
        return None, "Недостаточно данных"

    y = np.array(values[-n_last:])
    x = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    a = model.coef_[0]
    b = model.intercept_

    if a <= 0:
        return None, "Тренд не растёт"

    t_cross = (ucl - b) / a

    if t_cross <= n_last:
        return 0, "Уже вышел или выходит сейчас"

    steps_left = t_cross - n_last
    return steps_left, "OK"


def analyze_object_ucl_forecast(object_id):
    file_path = os.path.join(PATH, 'data_train', object_id)
    df = pd.read_csv(file_path)

    results = {}

    for gas in GAS_COLUMNS:
        series = df[gas].astype(float)
        cl, lcl, ucl = compute_control_limits(series)

        steps, status = predict_ucl_crossing(series.values, ucl)

        results[gas] = {
            "CL": cl,
            "UCL": ucl,
            "steps_to_UCL": steps,
            "status": status
        }

    return results


def main():
    y_data = pd.read_csv(os.path.join(PATH, 'train.csv'), index_col='id')

    example_ids = list(y_data.index[:5])  # только первые 5 объектов

    for object_id in example_ids:
        print(f"\nПрогноз выхода за UCL для объекта {object_id}")
        res = analyze_object_ucl_forecast(object_id)

        for gas, info in res.items():
            print(f"  Газ {gas}:")
            print(f"    UCL = {info['UCL']:.5f}")
            print(f"    Статус: {info['status']}")
            print(f"    Шагов до выхода: {info['steps_to_UCL']}\n")


if __name__ == '__main__':
    main()
