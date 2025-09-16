from pydantic import BaseModel

class MatchData(BaseModel):
    Team_A_avg_win_percentage: float
    Team_A_avg_KR: float
    Team_A_avg_elo: float
    Team_B_avg_win_percentage: float
    Team_B_avg_KR: float
    Team_B_avg_elo: float