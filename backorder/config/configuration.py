import os
from datetime import datetime
from pathlib import Path

from backorder import constant as C
from backorder.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from backorder.io import read_yaml_file
from backorder.logger import logging


class Configuration:
    def __init__(
        self,
        config_file_path: Path = C.CONFIG_FILE_PATH,
        current_time_stamp: str = C.CURRENT_TIME_STAMP,
    ) -> None:
        self.config_info = read_yaml_file(file_path=config_file_path)
        self.training_pipeline_config = self.get_training_pipeline_config()
        self.time_stamp = current_time_stamp

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        artifact_dir = self.training_pipeline_config.artifact_dir
        data_ingestion_info = self.config_info[C.DATA_INGESTION_CONFIG_KEY]

        data_ingestion_config = DataIngestionConfig(
            dataset_download_url=data_ingestion_info[C.DATA_INGESTION_DOWNLOAD_URL_KEY],
            tgz_download_dir=artifact_dir
            / C.DATA_INGESTION_ARTIFACT_DIR
            / self.time_stamp
            / data_ingestion_info[C.DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY],
            raw_data_dir=artifact_dir
            / C.DATA_INGESTION_ARTIFACT_DIR
            / self.time_stamp
            / data_ingestion_info[C.DATA_INGESTION_RAW_DATA_DIR_KEY],
            ingested_train_dir=artifact_dir
            / C.DATA_INGESTION_ARTIFACT_DIR
            / self.time_stamp
            / data_ingestion_info[C.DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            / data_ingestion_info[C.DATA_INGESTION_TRAIN_DIR_KEY],
            ingested_test_dir=artifact_dir
            / C.DATA_INGESTION_ARTIFACT_DIR
            / self.time_stamp
            / data_ingestion_info[C.DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            / data_ingestion_info[C.DATA_INGESTION_TEST_DIR_KEY],
        )
        logging.info(f"Data Ingestion config: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        artifact_dir = self.training_pipeline_config.artifact_dir

        data_validation_artifact_dir = os.path.join(
            artifact_dir, C.DATA_VALIDATION_ARTIFACT_DIR_NAME, self.time_stamp
        )
        data_validation_config = self.config_info[C.DATA_VALIDATION_CONFIG_KEY]

        data_validation_config = DataValidationConfig(
            schema_file_path=C.ROOT_DIR
            / data_validation_config[C.DATA_VALIDATION_SCHEMA_DIR_KEY]
            / data_validation_config[C.DATA_VALIDATION_SCHEMA_FILE_NAME_KEY],
            report_file_path=data_validation_artifact_dir
            / data_validation_config[C.DATA_VALIDATION_REPORT_FILE_NAME_KEY],
            report_page_file_path=data_validation_artifact_dir
            / data_validation_config[C.DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY],
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_config_info = self.config_info[C.DATA_TRANSFORMATION_CONFIG_KEY]

        data_transformation_config = DataTransformationConfig(
            preprocessed_object_file_path=self.training_pipeline_config.artifact_dir
            / C.DATA_TRANSFORMATION_ARTIFACT_DIR
            / self.time_stamp
            / data_transformation_config_info[C.DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
            / data_transformation_config_info[C.DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY],
            transformed_train_dir=self.training_pipeline_config.artifact_dir
            / C.DATA_TRANSFORMATION_ARTIFACT_DIR
            / self.time_stamp
            / data_transformation_config_info[C.DATA_TRANSFORMATION_DIR_NAME_KEY]
            / data_transformation_config_info[C.DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY],
            transformed_test_dir=self.training_pipeline_config.artifact_dir
            / C.DATA_TRANSFORMATION_ARTIFACT_DIR
            / self.time_stamp
            / data_transformation_config_info[C.DATA_TRANSFORMATION_DIR_NAME_KEY]
            / data_transformation_config_info[C.DATA_TRANSFORMATION_TEST_DIR_NAME_KEY],
        )

        logging.info(f"Data transformation config: {data_transformation_config}")
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        artifact_dir = self.training_pipeline_config.artifact_dir
        model_trainer_config_info = self.config_info[C.MODEL_TRAINER_CONFIG_KEY]

        model_trainer_config = ModelTrainerConfig(
            trained_model_file_path=artifact_dir
            / C.MODEL_TRAINER_ARTIFACT_DIR
            / self.time_stamp
            / model_trainer_config_info[C.MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
            / model_trainer_config_info[C.MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY],
            base_accuracy=model_trainer_config_info[C.MODEL_TRAINER_BASE_ACCURACY_KEY],
            model_config_file_path=model_trainer_config_info[C.MODEL_TRAINER_MODEL_CONFIG_DIR_KEY]
            / model_trainer_config_info[C.MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY],
        )
        logging.info(f"Model trainer config: {model_trainer_config}")
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        model_evaluation_config = self.config_info[C.MODEL_EVALUATION_CONFIG_KEY]
        response = ModelEvaluationConfig(
            model_evaluation_file_path=self.training_pipeline_config.artifact_dir
            / C.MODEL_EVALUATION_ARTIFACT_DIR
            / model_evaluation_config[C.MODEL_EVALUATION_FILE_NAME_KEY],
            time_stamp=self.time_stamp,
        )
        logging.info(f"Model Evaluation Config: {response}.")
        return response

    def get_model_pusher_config(
        self,
    ) -> ModelPusherConfig:
        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_pusher_config_info = self.config_info[C.MODEL_PUSHER_CONFIG_KEY]

        model_pusher_config = ModelPusherConfig(
            export_dir_path=C.ROOT_DIR
            / model_pusher_config_info[C.MODEL_PUSHER_MODEL_EXPORT_DIR_KEY]
            / time_stamp
        )
        logging.info(f"Model pusher config {model_pusher_config}")
        return model_pusher_config

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        training_pipeline_config = self.config_info[C.TRAINING_PIPELINE_CONFIG_KEY]
        training_pipeline_config = TrainingPipelineConfig(
            artifact_dir=C.ROOT_DIR
            / training_pipeline_config[C.TRAINING_PIPELINE_NAME_KEY]
            / training_pipeline_config[C.TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
        )
        logging.info(f"Training pipeline config: {training_pipeline_config}")
        return training_pipeline_config
