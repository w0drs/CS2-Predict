class DataLoadError(Exception):
    """Исключение для ошибок загрузки данных"""
    def __init__(self, message: str ="Ошибка при загрузке данных"):
        self.message = message
        super().__init__(self.message)

class FileReadError(Exception):
    """Исключение для ошибок чтения данных"""
    def __init__(self, message: str ="Ошибка при чтении файла с данными"):
        self.message = message
        super().__init__(self.message)

class DataReplaceError(Exception):
    """Исключение для ошибок замены в данных"""
    def __init__(self, message: str ="Ошибка при замене запятых на точки в данных"):
        self.message = message
        super().__init__(self.message)

class DumpError(Exception):
    """Исключение для ошибок замены в данных"""
    def __init__(
            self,
            message: str ="Ошибка при сохранении файла",
    ):
        self.message = message
        super().__init__(self.message)

class DataRemoveError(Exception):
    """Исключение для ошибок удаления строк в данных"""
    def __init__(self, message: str ="Ошибка при удалении в данных строк с определенными значениями"):
        self.message = message
        super().__init__(self.message)

class DataDropError(Exception):
    """Исключение для ошибок удаления колонок данных"""
    def __init__(self, message: str ="Ошибка при удалении определенных колонок в данных"):
        self.message = message
        super().__init__(self.message)

class DataProcessingError(Exception):
    """Исключение для ошибок при обработке данных"""
    def __init__(self, message: str ="Ошибка при обработке данных"):
        self.message = message
        super().__init__(self.message)

class FileLoadError(Exception):
    """Исключение для ошибок при обработке данных"""
    def __init__(self, message: str ="Ошибка при загрузке файла"):
        self.message = message
        super().__init__(self.message)

class DataTransformationError(Exception):
    """Исключение для ошибок трансформации данных"""
    def __init__(self, message: str = "Ошибка трансформации данных"):
        self.message = message
        super().__init__(self.message)

class ValidationError(Exception):
    """Исключение для ошибок валидации данных"""
    def __init__(self, message: str ="Ошибка валидации данных"):
        self.message = message
        super().__init__(self.message)