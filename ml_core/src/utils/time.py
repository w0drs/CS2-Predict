from datetime import datetime

def get_time() -> str:
    """
    Возвращает текущее время
    """
    now = datetime.now()
    return str(now.strftime("%Y-%m-%d_%H-%M-%S"))