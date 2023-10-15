from dataclasses import dataclass
from pathlib import Path
from sys import exc_info
from typing import Optional

import pandas as pd

from backorder.exception import BackorderException
from backorder.io import load_object


@dataclass
class BackorderData:
    national_inv: float
    lead_time: float
    in_transit_qty: float
    forecast_3_month: float
    forecast_6_month: float
    forecast_9_month: float
    sales_1_month: float
    sales_3_month: float
    sales_6_month: float
    sales_9_month: float
    min_bank: float
    potential_issue: object
    pieces_past_due: float
    perf_6_month_avg: float
    perf_12_month_avg: float
    local_bo_qty: float
    deck_risk: object
    oe_constraint: object
    ppap_risk: object
    stop_auto_buy: object
    rev_stop: object
    went_on_backorder: object = None

    def get_backorder_input_data_frame(self):
        backorder_input_dict = self.get_backorder_data_as_dict()
        return pd.DataFrame(backorder_input_dict)

    def get_backorder_data_as_dict(self):
        input_data = vars(self).copy()
        del input_data["went_on_backorder"]  # Remove 'went_on_backorder' from the dict
        return {k: [v] for k, v in input_data.items()}


class BackorderPredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def get_latest_model_path(self) -> Optional[Path]:
        model_folders = list(self.model_dir.iterdir())
        if not model_folders:
            return None

        latest_model_folder = max(map(str, model_folders))
        latest_model_dir = self.model_dir / f"{latest_model_folder}"

        model_files = list(latest_model_dir.iterdir())
        if not model_files:
            return None

        latest_model_file = model_files[0]
        latest_model_path = latest_model_dir / latest_model_file
        return latest_model_path

    def predict(self, X):
        model_path = self.get_latest_model_path()
        if model_path is None:
            raise BackorderException("No models found in the model directory", exc_info())

        model = load_object(file_path=model_path)
        went_on_backorder = model.predict(X)
        return went_on_backorder
