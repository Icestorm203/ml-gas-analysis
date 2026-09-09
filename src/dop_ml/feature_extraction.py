from tsfresh.feature_extraction import extract_features, MinimalFCParameters


def extract_tsfresh_features(X, y_data):
    """
    Извлекает признаки TSFresh (MinimalFCParameters).
    Возвращает матрицу признаков, отсортированную по id.
    """
    settings = MinimalFCParameters()

    X_feat = extract_features(
        X,
        column_id="id",
        default_fc_parameters=settings,
        n_jobs=0
    )

    X_feat = X_feat.loc[y_data.index]
    return X_feat
