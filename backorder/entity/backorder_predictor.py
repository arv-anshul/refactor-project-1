import os
from sys import exc_info
from typing import Optional

import pandas as pd

from backorder.exception import BackorderException
from backorder.util.util import load_object


class BackorderData:
    def __init__(
        self,
        national_inv: float,
        lead_time: float,
        in_transit_qty: float,
        forecast_3_month: float,
        forecast_6_month: float,
        forecast_9_month: float,
        sales_1_month: float,
        sales_3_month: float,
        sales_6_month: float,
        sales_9_month: float,
        min_bank: float,
        potential_issue: object,
        pieces_past_due: float,
        perf_6_month_avg: float,
        perf_12_month_avg: float,
        local_bo_qty: float,
        deck_risk: object,
        oe_constraint: object,
        ppap_risk: object,
        stop_auto_buy: object,
        rev_stop: object,
        went_on_backorder: object = None,
    ):
        self.national_inv = national_inv
        self.lead_time = lead_time
        self.in_transit_qty = in_transit_qty
        self.forecast_3_month = forecast_3_month
        self.forecast_6_month = forecast_6_month
        self.forecast_9_month = forecast_9_month
        self.sales_1_month = sales_1_month
        self.sales_3_month = sales_3_month
        self.sales_6_month = sales_6_month
        self.sales_9_month = sales_9_month
        self.min_bank = min_bank
        self.potential_issue = potential_issue
        self.pieces_past_due = pieces_past_due
        self.perf_6_month_avg = perf_6_month_avg
        self.perf_12_month_avg = perf_12_month_avg
        self.local_bo_qty = local_bo_qty
        self.deck_risk = deck_risk
        self.oe_constraint = oe_constraint
        self.ppap_risk = ppap_risk
        self.stop_auto_buy = stop_auto_buy
        self.rev_stop = rev_stop
        self.went_on_backorder = went_on_backorder

    def get_backorder_input_data_frame(self):
        backorder_input_dict = self.get_backorder_data_as_dict()
        return pd.DataFrame(backorder_input_dict)

    def get_backorder_data_as_dict(self):
        input_data = vars(self).copy()
        del input_data["went_on_backorder"]  # Remove 'went_on_backorder' from the dict
        return {k: [v] for k, v in input_data.items()}


class BackorderPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def get_latest_model_path(self) -> Optional[str]:
        model_folders = os.listdir(self.model_dir)
        if not model_folders:
            return None

        latest_model_folder = max(model_folders, key=int)
        latest_model_dir = os.path.join(self.model_dir, latest_model_folder)

        model_files = os.listdir(latest_model_dir)
        if not model_files:
            return None

        latest_model_file = model_files[0]
        latest_model_path = os.path.join(latest_model_dir, latest_model_file)
        return latest_model_path

    def predict(self, X):
        model_path = self.get_latest_model_path()
        if model_path is None:
            raise BackorderException("No models found in the model directory", exc_info())

        model = load_object(file_path=model_path)
        went_on_backorder = model.predict(X)
        return went_on_backorder
