import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH = os.path.join(BASE_DIR, 'data') + '\\'

GAS_COLUMNS = ['H2', 'CO', 'C2H4', 'C2H2']


def compute_control_limits(series: pd.Series):
    mu = series.mean()
    sigma = series.std(ddof=1)
    cl = mu
    ucl = mu + 3 * sigma
    lcl = mu - 3 * sigma
    return cl, lcl, ucl


def plot_shewhart_chart(time_index,
                        values,
                        cl,
                        lcl,
                        ucl,
                        title,
                        save_path=None):

    plt.figure(figsize=(9, 4))
    plt.plot(time_index, values, marker='o', linestyle='-', label='Наблюдения')

    plt.axhline(cl, color='green', linestyle='--', label=f'CL = {cl:.4f}')
    plt.axhline(ucl, color='red', linestyle='--', label=f'UCL = {ucl:.4f}')
    plt.axhline(lcl, color='red', linestyle='--', label=f'LCL = {lcl:.4f}')

    values = np.array(values)
    time_index = np.array(time_index)
    out_of_control = (values > ucl) | (values < lcl)
    if out_of_control.any():
        plt.scatter(time_index[out_of_control],
                    values[out_of_control],
                    color='red',
                    zorder=5,
                    label='Точки вне контроля')

    plt.title(title)
    plt.xlabel('Номер измерения')
    plt.ylabel('Концентрация, отн. ед.')
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def build_shewhart_for_object(object_id: str):

    file_path = os.path.join(PATH, 'data_train', object_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Файл для id={object_id} не найден: {file_path}')

    df = pd.read_csv(file_path)

    time_index = np.arange(len(df))

    stats = {}

    for col in GAS_COLUMNS:
        if col not in df.columns:
            print(f'Предупреждение: столбец {col} отсутствует в файле {file_path}')
            continue

        series = df[col].astype(float)

        cl, lcl, ucl = compute_control_limits(series)
        values = series.values

        out_of_control = (values > ucl) | (values < lcl)
        n_out = int(out_of_control.sum())
        stats[col] = {
            'CL': cl,
            'LCL': lcl,
            'UCL': ucl,
            'n_out_of_control': n_out
        }

        title = f'Карта Шухарта для {col}, объект {object_id}'
        save_dir = os.path.join(PATH, 'shewhart_charts', object_id)
        save_path = os.path.join(save_dir, f'shewhart_{col}.png')

        plot_shewhart_chart(time_index,
                            values,
                            cl,
                            lcl,
                            ucl,
                            title,
                            save_path=save_path)

    return stats


def main():
    y_data = pd.read_csv(os.path.join(PATH, 'train.csv'), index_col='id')

    example_ids = list(y_data.index[:5])  # только первые 5 объектов

    all_stats = {}

    for object_id in example_ids:
        print(f'Строим карты Шухарта для {object_id}')
        stats = build_shewhart_for_object(object_id)
        all_stats[object_id] = stats

    for object_id, obj_stats in all_stats.items():
        print(f'\nОбъект: {object_id}')
        for gas, st in obj_stats.items():
            print(f'  Газ {gas}: '
                  f'CL={st["CL"]:.4f}, LCL={st["LCL"]:.4f}, UCL={st["UCL"]:.4f}, '
                  f'точек вне контроля={st["n_out_of_control"]}')


if __name__ == '__main__':
    main()
