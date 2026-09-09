import pandas as pd

PATH = PATH = 'C:\\Users\\MSI\\Documents\\Cursor\\PythonProjects\\ml-gas-analysis\\data\\'


def load_data():
    """
    Загрузка целевых значений и временных рядов по каждому объекту.
    Возвращает:
      X  — объединённый DataFrame с временными рядами и колонкой id,
      y_data — DataFrame с целевыми значениями (index=id).
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
