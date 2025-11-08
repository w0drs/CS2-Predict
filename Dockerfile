FROM python:3.11-slim

WORKDIR /app

# Копируем requirements для кэширования
COPY requirements/api.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r api.txt

# Копируем нужные файлы проекта
COPY ml_core/ ./ml_core/
COPY configs/ ./configs/
COPY trained_models/ ./trained_models/
COPY scaler/ ./scaler/
COPY api/ ./api/

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.service:app", "--host", "0.0.0.0", "--port", "8000"]