import logging
import os

import pandas as pd

from backorder.constant import get_current_time_stamp


def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = get_log_file_name()
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",
    format="[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s",
    level=logging.INFO,
)


def get_log_dataframe(file_path):
    data = []
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    columns = ["Time stamp", "Log Level", "line number", "file name", "function name", "message"]
    log_df = pd.DataFrame(data, columns=columns)

    log_df["log_message"] = log_df["Time stamp"].astype(str) + ":$" + log_df["message"]
    return log_df[["log_message"]]
