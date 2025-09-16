DATASET_PATH: str = "data/raw/data_win_prediction.csv"
SCALERS_FOLDER_PATH: str = "scaler/"
MODELS_FOLDER_PATH: str = "trained_models/"
DATASET_SEP: str = ";"
DROP_COLUMN_LIST: list = ["\tMatch ID", "map"]
TARGET_COLUMN: str = "win"
REMOVE_COLUMN: str = 'map'
TARGET_REPLACE_CONDITION: dict = {"team a": 1, "team b": 0}
THRESHOLD: float = 0.823
RANDOM_STATE: int = 52
CAN_SAVE_MODELS: bool = False
MODEL_PARAMETERS: dict = {
        'C': 100,
        'penalty': 'l1',
        'solver': 'liblinear',
        'random_state': RANDOM_STATE,
    }
VALUES_TO_REPLACE: list = [
    "Team_A_avg_win_percentage",
    "Team_B_avg_win_percentage",
    "Team_A_avg_KR",
    "Team_A_avg_elo",
    "Team_B_avg_KR"
]
FEATURE_COLUMNS: list = [
    "Team_A_avg_win_percentage",
    "Team_A_avg_KR",
    "Team_A_avg_elo",
    "Team_B_avg_win_percentage",
    "Team_B_avg_KR",
    "Team_B_avg_elo"
]
VALUES_TO_REMOVE: list = [
    'aim_map',
    'awp_india',
    'aim_crashz_dust_1on1',
    'de_ravine',
    'de_foroglio',
    'awp_orange',
    'awp_lego_2',
    'aim_redline',
    'dorf'
]
