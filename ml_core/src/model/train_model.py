import joblib
from sklearn.linear_model import LogisticRegression

from ml_core.constants.constants import CAN_SAVE_MODELS
from ml_core.src.utils.time import get_time


def train_model(*, features, target, **kwargs):
    """
    Инициализация модели и ее обучение на переданных данных.
    Parameters:
        features: колонки с обработанными фичами
        target: целевая колонка
        kwargs: параметры для модели
    """
    model = LogisticRegression(**kwargs)
    model.fit(features, target)

    if CAN_SAVE_MODELS:
        joblib.dump(model, f'trained_models/model_{get_time()}.pkl')

    return model



