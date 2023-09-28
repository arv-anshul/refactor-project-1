import os
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import yaml

from backorder import constant as C


def write_yaml_file(file_path: Path, data: dict | None = None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as yaml_file:
        if data is not None:
            yaml.dump(data, yaml_file)


def read_yaml_file(file_path: Path) -> dict:
    with open(file_path, "rb") as yaml_file:
        return yaml.safe_load(yaml_file)


def save_numpy_array_data(file_path: Path, array: np.ndarray):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        np.save(file_obj, array)


def load_numpy_array_data(file_path: Path) -> np.ndarray:
    with open(file_path, "rb") as file_obj:
        return np.load(file_obj, allow_pickle=True)


def save_object(file_path: Path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)


def load_object(file_path: Path):
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)


def load_data(file_path: Path, schema_file_path: Path) -> pd.DataFrame:
    dataset_schema = read_yaml_file(schema_file_path)
    schema = dataset_schema[C.DATASET_SCHEMA_COLUMNS_KEY]
    dataframe = pd.read_csv(file_path)

    not_available_cols = []
    for column in dataframe.columns:
        if column in list(schema.keys()):
            dataframe[column].astype(schema[column])
        else:
            not_available_cols.append(column)

    if len(not_available_cols) > 0:
        raise ValueError(f"{not_available_cols!r} not present in the dataframe.")

    return dataframe
