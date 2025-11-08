from ml_core.src.utils.config_loader import load_config

def test_params():
    params = load_config("model_v1.yaml")
    if not isinstance(params, dict):
        raise FileNotFoundError