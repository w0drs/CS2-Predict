import yaml
from pathlib import Path

def load_config(config_name="model.yaml"):
    """
    Достает .yaml файл с параметрами модели (ну и не только)
    Parameters:
        config_name: файл с настройками модели
    """
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_model_params(config_name="model.yaml"):
    setting: dict = load_config(config_name)
    parameters = setting.get('model', {}).get('parameters', {})
    return parameters
