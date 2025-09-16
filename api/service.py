import joblib
import numpy as np
from fastapi import FastAPI

from ml_core.constants.constants import THRESHOLD
from api.data import MatchData
from ml_core.src.model.predict import predict_new_data
from ml_core.src.model.loader import load_scaler, load_model

app = FastAPI()
model = joblib.load(load_model())
scaler = joblib.load(load_scaler())

@app.post("/predict")
def predict(match: MatchData):
    features = [
        match.Team_A_avg_win_percentage,
        match.Team_A_avg_KR,
        match.Team_A_avg_elo,
        match.Team_B_avg_win_percentage,
        match.Team_B_avg_KR,
        match.Team_B_avg_elo
    ]
    prediction = predict_new_data(
        new_data=features,
        can_transform=False,
        use_scaler=True
    )[:, 1]
    proba = float(np.round(prediction, 3))
    score = 1 if proba > THRESHOLD else 0
    return {"score": score, "proba": proba}