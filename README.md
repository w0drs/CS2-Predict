# CS2 Faceit Match Winner Prediction
Полноценная система для предсказания победителей в CS2 матчах (Faceat) с REST API и веб-интерфейсом.

## 🎯 О проекте
Проект предсказывает вероятность победы команды на основе:
- Среднего K/D ratio команд
- Среднего Elo рейтинга  
- Среднего процента побед

Датасет был взят со следующего источника:
[Counter Strike 2 Win Prediction (FACEIT)](https://www.kaggle.com/datasets/piercehentosh/counter-strike-2-win-prediction-faceit)

## 📊 Метрики
ML модель достигает:
- *Precision*: 0.94
- *Recall*: 0.36
- *F1-Score*: 0.52

## 🌟 Возможности
- **REST API** на FastAPI
- **Веб-интерфейс** на Streamlit для визуализации
- **ML модель** с precision 94% для предсказания побед

## 🏗️ Структура проекта
```text
Cs2_predict/
├── api/                         # FastAPI микросервис
│   ├── data.py                  # класс данных для FastAPI приложения
│   └── service.py               # FastAPI приложение
├── app/                         # Streamlit фронтенд
│   └── app.py                   # Основное приложение
├── ml_core/                     # Ядро ML
│   ├── constants/               # Константные значения
│   │   └── constants.py  
│   ├── notebook/                # Код в jupiter notebook (файл с EDA и файл с обучением модели)
│   │   ├── 1.EDA.ipynb
│   │   └── 2.ML.ipynb
│   ├── src/                     # Модули для озагрузки датасета и обучения модели
│   │   ├── dataset/
│   │   │   └── make_dataset.py  # Загрузка датасета из файла и преобразование данных из него
│   │   ├── model/
│   │   │   ├── evaluate.py      # Проверка качества модели
│   │   │   ├── loader.py        # Модуль для загрузки модели и скейлера
│   │   │   ├── predict.py       # Модуль для предсказаний модели
│   │   │   └── train_model.py   # Модуль обучения модели
│   │   └── utils/
│   │       ├── exceptions.py    # Исключения для отладки кода
│   │       └── time.py          # Модуль работы с временем 
│   └── tests/                   # Тесты
│       └── unit_tests/
│           ├── test_dataset.py  # Тест загрузки датасета
│           └── test_loader.py   # Тесты для загрузчиков моделей и скейлеров
├── scaler/                      # .pkl файлы со скейлерами (MinMaxScaler)
├── trained_models/              # .pkl файлы с обученной моделью
├── main.py                      # Точка входа в программу обучения модели
├── pytest.ini
└── README.md
```

## Локальный запуск
```bash
# API сервер
uvicorn api.main:app --reload --port 8000

# Frontend
streamlit run frontend/app.py
```

## 🖥️ Веб-интерфейс
Streamlit приложение доступно по адресу: http://localhost:8501



