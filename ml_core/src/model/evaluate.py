from sklearn.metrics import classification_report, precision_score, recall_score

from ml_core.constants.constants import THRESHOLD, SCALERS_FOLDER_PATH, MODELS_FOLDER_PATH
from ml_core.src.model.predict import predict_new_data



def evaluate_model(X_test, y_test, can_transform: bool = False, use_scaler: bool = False):
    """Оценка обученной модели"""
    probabilities = predict_new_data(
        new_data=X_test,
        scaler_path=SCALERS_FOLDER_PATH,
        model_path=MODELS_FOLDER_PATH,
        can_transform=can_transform,
        use_scaler=use_scaler
    )
    y_scores = probabilities[:, 1]
    y_pred_high_precision = (y_scores >= THRESHOLD).astype(int)

    print(f"Precision: {precision_score(y_test, y_pred_high_precision):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_high_precision):.4f}")
    print(classification_report(y_test, y_pred_high_precision))

    return y_pred_high_precision