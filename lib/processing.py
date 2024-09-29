import io
import re
from typing import List, Match, Optional

import pandas as pd


def check_none(description: pd.Series) -> pd.Series:
    """Проверка на пустые строки"""

    if description.isna().sum() > 0:
        return description.fillna("Нет описания")
    return description


def predict_proba(text: str, pipeline) -> float:
    """Получение вероятности контакта в тексте"""

    LABEL_0 = "LABEL_0"
    predict = pipeline(text)[0]
    if predict["label"] == LABEL_0:
        return 1 - predict["score"]
    return predict["score"]


def get_regular(text: str, regular: str) -> List[Match[str]]:
    """Получение списка с результатами поиска паттернов"""

    regex_split = regular.split(";")
    result = [list(re.finditer(reg, text)) for reg in regex_split]
    return sum(result, [])


class TqdmToLogger(io.StringIO):
    """Логирование stream для tqdm"""

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)
