import joblib
import numpy as np
import pandas as pd
from ml_core.constants.constants import VALUES_TO_REPLACE, DROP_COLUMN_LIST, SCALERS_FOLDER_PATH, MODELS_FOLDER_PATH, \
    FEATURE_COLUMNS
from ml_core.src.model.loader import load_scaler, load_model
from ml_core.src.utils.exceptions import DataTransformationError, FileLoadError

from typing import Optional
from sklearn.linear_model import LogisticRegression


def predict_new_data(
        *,
        new_data: pd.DataFrame | list,
        scaler_path: str | None = SCALERS_FOLDER_PATH,
        model_path: str | None = MODELS_FOLDER_PATH,
        can_transform: bool = True,
        use_scaler: bool = True,
):
    """
    Предсказания для новых данных с обработкой
    """
    try:
        trained_model: Optional[LogisticRegression] | None = None
        if isinstance(new_data, list):
            data_2d = np.array(new_data).reshape(1, -1)
            df = pd.DataFrame(data=data_2d, columns=FEATURE_COLUMNS)
        else:
            df = new_data.copy()

        if model_path:
            last_model_path = load_model(model_path)
            trained_model = joblib.load(last_model_path)
        else:
            raise FileLoadError(f"Ошибка при загрузке файла с моделью")

        if can_transform:
            # Применяем те же преобразования
            for col in VALUES_TO_REPLACE:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].str.replace(",", ".").astype(float)

            if DROP_COLUMN_LIST:
                df = df.drop(columns=[col for col in DROP_COLUMN_LIST if col in df.columns])

        # Загрузка и применение скейлера
        if scaler_path and use_scaler:
            last_scaler_path = load_scaler(scaler_path) # последний (по времени) файл с MinMaxScaler
            scaler = joblib.load(last_scaler_path)
            features = FEATURE_COLUMNS
            df[features] = scaler.transform(df[features])

        #predictions = trained_model.predict(df)
        probabilities = trained_model.predict_proba(df)

        return probabilities

    except Exception as e:
        raise DataTransformationError(f"Ошибка предсказания: {e}")
