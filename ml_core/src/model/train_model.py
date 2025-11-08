import joblib
from sklearn.linear_model import LogisticRegression

from ml_core.src.utils.time import get_time


def train_model(*, features, target, save_model=False, **kwargs):
    """
    Инициализация модели и ее обучение на переданных данных.
    Parameters:
        features: колонки с обработанными фичами
        target: целевая колонка
        save_model: сохранять ли обученную модель в файл
        kwargs: параметры для модели
    """
    model = LogisticRegression(**kwargs)
    model.fit(features, target)

    if save_model:
        joblib.dump(model, f'trained_models/model_{get_time()}.pkl')

    return model



