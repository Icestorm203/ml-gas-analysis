import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH = os.path.join(BASE_DIR, 'data') + '\\'


def load_data():
    """
    Загружает train.csv и временные ряды из data_train/.
    Возвращает:
      X — объединённый DataFrame с временными рядами и колонкой id,
      y_data — DataFrame с целевыми значениями.
    """
    y_data = pd.read_csv(PATH + 'train.csv', index_col='id')

    X_data = {}
    for file_name in y_data.index:
        path = PATH + f'data_train/{file_name}'
        X_data[file_name] = pd.read_csv(path)

    X = pd.concat(
        [X_data[file].assign(id=file) for file in y_data.index],
        axis=0,
        ignore_index=True
    )

    return X, y_data
