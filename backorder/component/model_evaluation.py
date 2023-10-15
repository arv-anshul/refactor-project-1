import os

import numpy as np

from backorder import constant as C
from backorder.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from backorder.entity.config_entity import ModelEvaluationConfig
from backorder.entity.model_factory import evaluate_classification_model
from backorder.io import load_data, load_object, read_yaml_file, write_yaml_file
from backorder.logger import logging


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact

    def get_best_model(self):
        model = None
        model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

        if not os.path.exists(
            model_evaluation_file_path
        ):  # it will run only at  the start when no model is there
            write_yaml_file(
                file_path=model_evaluation_file_path,
            )
            return model
        model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

        model_eval_file_content = (
            dict() if model_eval_file_content is None else model_eval_file_content
        )  # gets empty dict if none and if avilable directly pass that file

        if C.BEST_MODEL_KEY not in model_eval_file_content:
            return model

        model = load_object(file_path=model_eval_file_content[C.BEST_MODEL_KEY][C.MODEL_PATH_KEY])
        return model

    def update_evaluation_report(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ):  # function update/replace model when model acc > base model(model in production)
        eval_file_path = self.model_evaluation_config.model_evaluation_file_path
        model_eval_content = read_yaml_file(file_path=eval_file_path)
        model_eval_content = dict() if model_eval_content is None else model_eval_content

        previous_best_model = None
        if C.BEST_MODEL_KEY in model_eval_content:
            previous_best_model = model_eval_content[C.BEST_MODEL_KEY]

        logging.info(f"Previous eval result: {model_eval_content}")
        eval_result = {
            C.BEST_MODEL_KEY: {
                C.MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
            }
        }
        # creating history of updation of model
        if previous_best_model is not None:
            model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
            if C.HISTORY_KEY not in model_eval_content:
                history = {C.HISTORY_KEY: model_history}
                eval_result.update(history)
            else:
                model_eval_content[C.HISTORY_KEY].update(model_history)

        model_eval_content.update(eval_result)
        logging.info(f"Updated eval result:{model_eval_content}")
        write_yaml_file(file_path=eval_file_path, data=model_eval_content)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
        trained_model_object = load_object(file_path=trained_model_file_path)

        train_file_path = self.data_ingestion_artifact.train_file_path
        test_file_path = self.data_ingestion_artifact.test_file_path

        schema_file_path = self.data_validation_artifact.schema_file_path

        train_dataframe = load_data(
            file_path=train_file_path,
            schema_file_path=schema_file_path,
        )
        test_dataframe = load_data(
            file_path=test_file_path,
            schema_file_path=schema_file_path,
        )
        schema_content = read_yaml_file(file_path=schema_file_path)
        target_column_name = schema_content[C.TARGET_COLUMN_KEY]

        # target_column
        logging.info("Converting target column into numpy array.")
        train_target_arr = np.array(train_dataframe[target_column_name])
        test_target_arr = np.array(test_dataframe[target_column_name])
        logging.info("Conversion completed target column into numpy array.")

        # dropping target column from the dataframe
        logging.info("Dropping target column from the dataframe.")
        train_dataframe.drop(target_column_name, axis=1, inplace=True)
        test_dataframe.drop(target_column_name, axis=1, inplace=True)
        logging.info("Dropping target column from the dataframe completed.")

        model = self.get_best_model()

        if model is None:
            logging.info("Not found any existing model. Hence accepting trained model")
            model_evaluation_artifact = ModelEvaluationArtifact(
                evaluated_model_path=trained_model_file_path, is_model_accepted=True
            )
            self.update_evaluation_report(model_evaluation_artifact)
            logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
            return model_evaluation_artifact

        model_list = [model, trained_model_object]  # old model and newly trained model

        metric_info_artifact = evaluate_classification_model(
            model_list=model_list,
            X_train=train_dataframe,
            y_train=train_target_arr,
            X_test=test_dataframe,
            y_test=test_target_arr,
            base_accuracy=self.model_trainer_artifact.model_accuracy,
        )
        logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

        if metric_info_artifact is None:  # if both model doesnt achieve base accuracy then
            response = ModelEvaluationArtifact(
                is_model_accepted=False, evaluated_model_path=trained_model_file_path
            )
            logging.info(response)
            return response

        if (
            metric_info_artifact.index_number == 1
        ):  # if model at index 1i.e trained model >base acc then
            model_evaluation_artifact = ModelEvaluationArtifact(
                evaluated_model_path=trained_model_file_path, is_model_accepted=True
            )
            self.update_evaluation_report(model_evaluation_artifact)
            logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

        else:  # if trained model is < base acc no need to update
            logging.info(
                "Trained model is no better than existing model hence not accepting trained model"
            )
            model_evaluation_artifact = ModelEvaluationArtifact(
                evaluated_model_path=trained_model_file_path, is_model_accepted=False
            )
        return model_evaluation_artifact

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")
