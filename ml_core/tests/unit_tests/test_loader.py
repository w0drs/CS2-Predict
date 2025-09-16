from ml_core.src.model.loader import load_scaler, load_model
from contextlib import nullcontext as does_not_raise
import pytest

@pytest.mark.parametrize(
    argnames="folder_path, expectation",
    argvalues=[
        ("scaler/", does_not_raise()),
        ("empty_folder/", pytest.raises(FileNotFoundError)),
        ("/folder", pytest.raises(FileNotFoundError))
    ]
)
def test_scaler_load(folder_path: str, expectation):
    with expectation:
        scaler = load_scaler(folder_path)
        # проверка - существует ли файл
        assert scaler
        # проверка, оканчивается ли файл нужным нам форматом
        assert scaler.endswith(".pkl") or scaler.endswith(".joblib")

@pytest.mark.parametrize(
    argnames="folder_path, expectation",
    argvalues=[
        ("trained_models/", does_not_raise()),
        ("empty_folder/", pytest.raises(FileNotFoundError)),
        ("/folder", pytest.raises(FileNotFoundError))
    ]
)
def test_model_load(folder_path: str, expectation):
    with expectation:
        model = load_model(folder_path)
        # проверка - существует ли файл
        assert model
        # проверка, оканчивается ли файл нужным нам форматом
        assert model.endswith(".pkl") or model.endswith(".joblib")
