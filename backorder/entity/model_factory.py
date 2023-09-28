import importlib
import os
from sys import exc_info
from typing import Any, NamedTuple

import numpy as np
import yaml
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from backorder.exception import BackorderException
from backorder.logger import logging


class InitializedModelDetail(NamedTuple):
    model_serial_number: str
    model: str
    param_grid_search: str
    model_name: str


class GridSearchedBestModel(NamedTuple):
    model_serial_number: str
    model: str
    best_model: BaseEstimator
    best_parameters: dict[str, Any]
    best_score: float


class MetricInfoArtifact(NamedTuple):
    model_name: str
    model_object: str
    train_f1_score: float
    test_f1_score: float
    train_accuracy: float
    test_accuracy: float
    model_accuracy: float
    index_number: int


def evaluate_classification_model(
    model_list: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_accuracy: float,
) -> MetricInfoArtifact | None:
    index_number = 0
    metric_info_artifact = None
    for model in model_list:
        model_name = str(model)
        logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

        print(f"{model}")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1_score = f1_score(y_train, y_train_pred, average="macro")
        test_f1_score = f1_score(y_test, y_test_pred, average="macro")

        model_accuracy = float(
            (2 * (train_f1_score * test_f1_score)) / (train_f1_score + test_f1_score)
        )
        diff_test_train_acc = abs(test_f1_score - train_f1_score)

        logging.info(f"{'>>'*30} Score {'<<'*30}")
        logging.info("Train Score\t\t Test Score\t\t Average Score")
        logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

        logging.info(f"{'>>'*30} Loss {'<<'*30}")
        logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].")
        logging.info(f"Train f1_score : [{train_f1_score}].")
        logging.info(f"Test f1_score : [{test_f1_score}].")

        if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
            base_accuracy = model_accuracy
            metric_info_artifact = MetricInfoArtifact(
                model_name=model_name,
                model_object=model,
                train_f1_score=float(train_f1_score),
                test_f1_score=float(test_f1_score),
                train_accuracy=float(train_acc),
                test_accuracy=float(test_acc),
                model_accuracy=model_accuracy,
                index_number=index_number,
            )

            logging.info(f"Acceptable model found {metric_info_artifact}. ")
        index_number += 1

    if metric_info_artifact is None:
        logging.warning("No model found with higher accuracy than base accuracy")

    return metric_info_artifact


def get_sample_model_config_yaml_file(export_dir: str):
    model_config = {
        "grid_search": {
            "module": "sklearn.model_selection",
            "class": "GridSearchCV",
            "params": {"cv": 5, "verbose": 3},
        },
        "model_selection": {
            "module_0": {
                "module": "module_of_model",
                "class": "ModelClassName",
                "params": {
                    "param_name1": "value1",
                    "param_name2": "value2",
                },
                "search_param_grid": {"param_name": ["param_value_1", "param_value_2"]},
            },
        },
    }

    os.makedirs(export_dir, exist_ok=True)
    export_file_path = os.path.join(export_dir, "model.yaml")
    with open(export_file_path, "w") as file:
        yaml.dump(model_config, file)

        return export_file_path


class ModelFactory:
    def __init__(self, model_config_path: str):
        self.config = self._read_params(model_config_path)
        self.grid_search_module = self.config["grid_search"]["module"]
        self.grid_search_class = self.config["grid_search"]["class"]
        self.grid_search_params = self.config["grid_search"]["params"]
        self.models_config = self.config["model_selection"]
        self.initialized_model_list = None
        self.grid_searched_best_model_list = None

    @staticmethod
    def _update_properties(instance, properties):
        for key, value in properties.items():
            setattr(instance, key, value)
        return instance

    @staticmethod
    def _read_params(config_path: str):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    @staticmethod
    def _get_class(module_name: str, class_name: str):
        module = importlib.import_module(module_name)
        class_ref = getattr(module, class_name)
        return class_ref

    def _execute_grid_search(
        self, initialized_model: InitializedModelDetail, input_feature, output_feature
    ):
        grid_search_class = self._get_class(self.grid_search_module, self.grid_search_class)
        grid_search: GridSearchCV = grid_search_class(
            estimator=initialized_model.model, param_grid=initialized_model.param_grid_search
        )
        grid_search: GridSearchCV = self._update_properties(grid_search, self.grid_search_params)
        grid_search.fit(input_feature, output_feature)
        grid_searched_best_model = GridSearchedBestModel(
            model_serial_number=initialized_model.model_serial_number,
            model=initialized_model.model,
            best_model=grid_search.best_estimator_,
            best_parameters=grid_search.best_params_,
            best_score=grid_search.best_score_,
        )
        return grid_searched_best_model

    def get_initialized_model_list(self) -> list[InitializedModelDetail]:
        initialized_model_list = []
        for serial_number, model_config in self.models_config.items():
            model_class = self._get_class(model_config["module"], model_config["class"])
            model = model_class()

            if "params" in model_config:
                model_params = model_config["params"]
                model = self._update_properties(model, model_params)

            param_grid_search = model_config["search_param_grid"]
            model_name = f"{model_config['module']}.{model_config['class']}"
            initialized_model = InitializedModelDetail(
                model_serial_number=serial_number,
                model=model,
                param_grid_search=param_grid_search,
                model_name=model_name,
            )
            initialized_model_list.append(initialized_model)

        self.initialized_model_list = initialized_model_list
        return self.initialized_model_list

    def _initiate_best_parameter_search(self, initialized_model, input_feature, output_feature):
        return self._execute_grid_search(
            initialized_model=initialized_model,
            input_feature=input_feature,
            output_feature=output_feature,
        )

    def initiate_best_parameter_search_for_initialized_models(
        self, initialized_model_list: list[InitializedModelDetail], input_feature, output_feature
    ) -> list[GridSearchedBestModel]:
        self.grid_searched_best_model_list = []
        for initialized_model in initialized_model_list:
            grid_searched_best_model = self._initiate_best_parameter_search(
                initialized_model=initialized_model,
                input_feature=input_feature,
                output_feature=output_feature,
            )
            self.grid_searched_best_model_list.append(grid_searched_best_model)
        return self.grid_searched_best_model_list

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(
        grid_searched_best_model_list: list[GridSearchedBestModel], base_accuracy=0.45
    ) -> GridSearchedBestModel:
        best_model = None
        for grid_searched_best_model in grid_searched_best_model_list:
            if base_accuracy < grid_searched_best_model.best_score:
                base_accuracy = grid_searched_best_model.best_score
                best_model = grid_searched_best_model
        if not best_model:
            raise BackorderException(
                "None of the models has the required base accuracy", exc_info()
            )
        return best_model

    def get_best_model(self, X, y, base_accuracy=0.45) -> GridSearchedBestModel:
        initialized_model_list = self.get_initialized_model_list()
        grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
            initialized_model_list=initialized_model_list, input_feature=X, output_feature=y
        )
        return self.get_best_model_from_grid_searched_best_model_list(
            grid_searched_best_model_list, base_accuracy=base_accuracy
        )
