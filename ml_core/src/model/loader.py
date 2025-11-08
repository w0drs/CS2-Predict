from ml_core.src.utils.config_loader import load_config


def load_scaler(yaml_file: str = "model_v1.yaml") -> str:
    """
    Достает из yaml файла путь до скейлера и возвращает его.
    Raises:
        FileNotFoundError: Если файл не найден
    """
    model_settings: dict = load_config(yaml_file)
    scaler_path = model_settings.get("model",{}).get('paths', {}).get("scaler_file", None)
    if not scaler_path:
        raise FileNotFoundError(f"Файл скейлера не найден")
    return str(scaler_path)


def load_model(yaml_file: str = "model_v1.yaml") -> str:
    """
    Достает из yaml файла путь до обученной модели и возвращает его.
    Raises:
        FileNotFoundError: Если файл не найден
    """
    model_settings: dict = load_config(yaml_file)
    model_path = model_settings.get("model", {}).get('paths', {}).get("model_file", None)
    if not model_path:
        raise FileNotFoundError(f"Файл модели не найден")
    return str(model_path)
