from ml_core.src.model.loader import load_scaler, load_model
from contextlib import nullcontext as does_not_raise
import pytest

@pytest.mark.parametrize(
    argnames="yaml_file, expectation",
    argvalues=[
        ("model_v1.yaml", does_not_raise()),
        ("model.yaml", pytest.raises(FileNotFoundError)),
        ("", pytest.raises(PermissionError))
    ]
)
def test_model_load(yaml_file: str, expectation):
    with expectation:
        scaler = load_scaler(yaml_file)
        # проверка - существует ли файл
        assert scaler
        # проверка, оканчивается ли файл нужным нам форматом
        assert scaler.endswith(".pkl") or scaler.endswith(".joblib")

@pytest.mark.parametrize(
    argnames="yaml_file, expectation",
    argvalues=[
        ("model_v1.yaml", does_not_raise()),
        ("model.yaml", pytest.raises(FileNotFoundError)),
        ("", pytest.raises(PermissionError))
    ]
)
def test_model_load(yaml_file: str, expectation):
    with expectation:
        model = load_model(yaml_file)
        # проверка - существует ли файл
        assert model
        # проверка, оканчивается ли файл нужным нам форматом
        assert model.endswith(".pkl") or model.endswith(".joblib")
