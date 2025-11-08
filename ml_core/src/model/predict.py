import joblib
import numpy as np
import pandas as pd
from ml_core.constants.constants import VALUES_TO_REPLACE, DROP_COLUMN_LIST, FEATURE_COLUMNS
from ml_core.src.utils.exceptions import DataTransformationError, FileLoadError

from typing import Optional
from sklearn.linear_model import LogisticRegression


def predict_new_data(
        *,
        new_data: pd.DataFrame | list,
        loaded_model=None,
        loaded_scaler=None,
        scaler_path: str | None = None,
        model_path: str | None = None,
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
            trained_model = joblib.load(model_path)
        elif loaded_model is not None:
            trained_model = loaded_model
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
            scaler = None
            if scaler_path:
                scaler = joblib.load(scaler_path)
            elif loaded_scaler is not None:
                scaler = loaded_scaler
            features = FEATURE_COLUMNS
            df[features] = scaler.transform(df[features])

        #predictions = trained_model.predict(df)
        probabilities = trained_model.predict_proba(df)

        return probabilities

    except Exception as e:
        raise DataTransformationError(f"Ошибка предсказания: {e}")
