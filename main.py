from ml_core.src.dataset.make_dataset import get_and_split_dataset
from ml_core.src.model.evaluate import evaluate_model
from ml_core.src.model.train_model import train_model
from ml_core.src.utils.config_loader import load_model_params
from ml_core.src.utils.exceptions import DataLoadError


def main():
    """
    Главный файл с обучением и сохранением модели и выведением метрик
    """
    config_file = 'model_v1.yaml'
    # разделение данных для обучения и тестирования
    X_train, X_test, y_train, y_test = get_and_split_dataset(save_scaler=False)

    # выводим ошибку (если есть)
    if X_train is None:
        raise DataLoadError

    model_params = load_model_params(config_file)

    # обучение модели
    trained_model = train_model(
        features=X_train,
        target=y_train,
        save_model=False,
        **model_params
    )

    # оценка качества модели
    evaluate_model(
        X_test=X_test,
        y_test=y_test,
        can_transform=False,
        use_scaler=False,
        yaml_file=config_file
    )

if __name__ == "__main__":
    main()