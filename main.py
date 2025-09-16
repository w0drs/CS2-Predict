from ml_core.constants.constants import MODEL_PARAMETERS
from ml_core.src.dataset.make_dataset import get_and_split_dataset
from ml_core.src.model.evaluate import evaluate_model
from ml_core.src.model.train_model import train_model
from ml_core.src.utils.exceptions import DataLoadError


def main():
    # разделение данных для обучения и тестирования
    X_train, X_test, y_train, y_test = get_and_split_dataset()

    # выводим ошибку (если есть)
    if X_train is None:
        raise DataLoadError

    # обучение модели
    trained_model = train_model(
        features=X_train,
        target=y_train,
        **MODEL_PARAMETERS
    )

    # оценка качества модели
    evaluate_model(
        X_test=X_test,
        y_test=y_test,
        can_transform=False,
        use_scaler=False
    )

if __name__ == "__main__":
    main()