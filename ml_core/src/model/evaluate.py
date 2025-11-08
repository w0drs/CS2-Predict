import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score

from ml_core.constants.constants import THRESHOLD
from ml_core.src.model.loader import load_model, load_scaler
from ml_core.src.model.predict import predict_new_data


def evaluate_model(
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame | pd.Series,
        can_transform: bool = False,
        use_scaler: bool = False,
        yaml_file: str = 'model_v1.yaml'
):
    """Оценка обученной модели"""
    model_path = load_model(yaml_file=yaml_file)
    scaler_path = load_scaler(yaml_file=yaml_file)

    probabilities = predict_new_data(
        new_data=X_test,
        scaler_path=scaler_path,
        model_path=model_path,
        can_transform=can_transform,
        use_scaler=use_scaler
    )
    y_scores = probabilities[:, 1]
    y_pred_high_precision = (y_scores >= THRESHOLD).astype(int)

    print(f"Precision: {precision_score(y_test, y_pred_high_precision):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_high_precision):.4f}")
    print(classification_report(y_test, y_pred_high_precision))

    return y_pred_high_precision