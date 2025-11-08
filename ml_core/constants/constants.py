DATASET_PATH: str = "ml_core/raw/data_win_prediction.csv"
DATASET_SEP: str = ";"
DROP_COLUMN_LIST: list = ["\tMatch ID", "map"]
TARGET_COLUMN: str = "win"
REMOVE_COLUMN: str = 'map'
TARGET_REPLACE_CONDITION: dict = {"team a": 1, "team b": 0}
THRESHOLD: float = 0.5608
RANDOM_STATE: int = 42
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
