from pathlib import Path

from ml_core.constants.constants import SCALERS_FOLDER_PATH, MODELS_FOLDER_PATH


def load_scaler(folder_path: str = SCALERS_FOLDER_PATH) -> str:
    """
    Загружает из папки scaler последний файл с MinMaxScaler.
    Возвращает путь до этого скейлера.

    Raises:
        FileNotFoundError: Если папка не существует или пуста
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка {folder_path} не существует")

    files = list(folder.iterdir())
    if not files:
        raise FileNotFoundError(f"Папка {folder_path} пуста")

    # Сортируем по времени изменения (последний созданный файл)
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)


def load_model(folder_path: str = MODELS_FOLDER_PATH) -> str:
    """
    Загружает из папки trained_models последний файл с обученной моделью.
    Возвращает путь до этой модели.

    Raises:
        FileNotFoundError: Если папка не существует или пуста
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка {folder_path} не существует")

    files = list(folder.iterdir())
    if not files:
        raise FileNotFoundError(f"Папка {folder_path} пуста")

    # Сортируем по времени изменения (последний созданный файл)
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)