from typing import List, Tuple, Union

import pandas as pd
from config import CONTACT, device, max_length, model_path, truncation
from processing import TqdmToLogger, check_none, get_regular, predict_proba
from tqdm.auto import tqdm
from transformers import pipeline as pipe


def task1(description: pd.Series, tqdm_out: TqdmToLogger) -> List[float]:
    """Получение результатов предсказания модели поиска контактов в тексте"""

    description = check_none(description)
    clf = pipe(
        "text-classification",
        model=model_path,
        device=device,
        truncation=truncation,
        max_length=max_length,
    )
    dataset_pbar = tqdm(description, file=tqdm_out, mininterval=30)
    result = [predict_proba(data, clf) for data in dataset_pbar]
    return result


def task2(
    description: str, regular=CONTACT
) -> Union[Tuple[int, int], Tuple[None, None]]:
    """Получение результатов предсказания позиции контакта в строке"""

    result = get_regular(str(description), regular)
    if len(result) == 1:
        return result[0].start(), result[0].end() - 1
    return None, None
