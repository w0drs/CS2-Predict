import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml_core.constants.constants import (DATASET_SEP, DATASET_PATH, DROP_COLUMN_LIST, VALUES_TO_REMOVE,
                                      VALUES_TO_REPLACE, TARGET_COLUMN, REMOVE_COLUMN, TARGET_REPLACE_CONDITION,
                                      RANDOM_STATE, CAN_SAVE_MODELS)
from ml_core.src.utils.exceptions import (FileReadError, DataReplaceError, DataRemoveError, DataDropError,
                                       DataProcessingError, DumpError, DataTransformationError)
from ml_core.src.utils.time import get_time


def _load_dataset(
        *,
        path: str,
        sep: str = ","
) -> pd.DataFrame | None:
    """
    Загружает датасет из csv файла и возвращает его
    Parameters:
        path: путь до файла (вместе с его названием и форматом)
        sep: разделитель для загрузки csv файла
    """
    dataset = None
    try:
        dataset = pd.read_csv(path, sep=sep)
        return dataset
    except FileReadError:
        print("Ошибка при чтении файла с данными")
    finally:
        return dataset

def _transform_dataset(
        *,
        dataset: pd.DataFrame,
        scaler = None,
        drop_column_list: list = None,
        values_to_remove: list = None,
        remove_column: str = None,
        values_to_replace: list = None
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """
    Обработка данных: изменение типа данных и обработка по MinMaxScaler.
    Parameters:
        dataset: наш загруженный датасет (из функции _load_dataset())
        scaler: преобразователь данных (в этом проекте тут будет передаваться MinMaxScaler())
        drop_column_list: колонки, которые следует убрать (они не нужны). В этом проекте это "Match ID" и "map"
        values_to_remove: это список ненужных значений. Если они будут найдены в колонке 'map', то такую строку убираем
        remove_column: колонка ('map'). В ней мы будем искать ненужные значения (из списка values_to_remove)
        values_to_replace: колонки, в строчках которых мы будем заменять ',' на '.', чтобы преобразовать строки в числа
    """
    if dataset is None: return None, None

    df = dataset.copy()
    # Проверяем целевую колонку
    if TARGET_COLUMN not in df.columns:
        raise DataTransformationError(f"Целевая колонка '{TARGET_COLUMN}' не найдена")

    # убираем строчки, если в колонке 'map' есть ненужные для нас значения
    if values_to_remove:
        try:
            df = df[~df[remove_column].isin(values_to_remove)]
        except DataRemoveError:
            print("Ошибка при удалении в данных строк с определенными значениями")
            return None, None

    # Преобразуем target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_REPLACE_CONDITION)

    # Разделяем на X и y
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # убираем ненужные колонки
    if drop_column_list:
        try:
            X = X.drop(columns=drop_column_list)
        except DataDropError:
            print("Ошибка при удалении определенных колонок в данных")
            return None, None

    # заменяем ',' на '.'
    if values_to_replace:
        try:
            for col in values_to_replace:
                X[col] = X[col].str.replace(",",".").astype(float)
        except DataReplaceError:
            print("Ошибка при замене запятых на точки в данных")
            return None, None

    features = X.columns.to_list()

    # преобразуем колонки по MinMaxScaler
    if scaler:
        try:
            X[features] = scaler.fit_transform(X[features])
        except DataProcessingError:
            print("Ошибка при обработке данных")
            return None, None

    if CAN_SAVE_MODELS:
        try:
            joblib.dump(scaler, f'scaler/scaler_{get_time()}.pkl')
        except DumpError:
            print("Ошибка при сохранении настроек MinMaxScaler в файл")
            return None, None

    return X, y

def get_and_split_dataset(
        test_size: int = 0.2,
        random_state: int = RANDOM_STATE,
):
    """
    Весь процесс загрузки, преобразования и разделения датасета
    Parameters:
         test_size: размер для тестов выборки в долях
         random_state: число для воспроизводимости результата
    """
    scaler = MinMaxScaler()

    # загрузка датасета
    dataset = _load_dataset(
        path=DATASET_PATH,
        sep=DATASET_SEP
    )

    # обработка датасета и разделение его на колонки с фичами и на таргет колонку
    X, y = _transform_dataset(
        dataset=dataset,
        scaler=scaler,
        drop_column_list=DROP_COLUMN_LIST,
        remove_column=REMOVE_COLUMN,
        values_to_remove=VALUES_TO_REMOVE,
        values_to_replace=VALUES_TO_REPLACE,
    )

    if X is None:
        return [None, None, None, None]

    # разделение датасета
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)





