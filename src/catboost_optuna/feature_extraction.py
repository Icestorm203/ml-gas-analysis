from tsfresh.feature_extraction import extract_features, MinimalFCParameters


def extract_tsfresh_features(X, y_data):
    """
    Извлечение признаков TSFresh (MinimalFCParameters) из временных рядов.
    На вход:
      X      — DataFrame с временными рядами и колонкой id,
      y_data — DataFrame с индексом id (для фильтрации признаков).
    Возвращает:
      X_feat — матрицу признаков, отсортированную по id из y_data.
    """
    settings = MinimalFCParameters()
    X_feat = extract_features(
        X,
        column_id="id",
        default_fc_parameters=settings,
        n_jobs=4
    )
    X_feat = X_feat.loc[y_data.index]
    return X_feat
