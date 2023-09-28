import os
import uuid
from datetime import datetime
from pathlib import Path
from sys import exc_info
from threading import Thread
from typing import NamedTuple

import pandas as pd

from backorder import constant as C
from backorder.component.data_ingestion import DataIngestion
from backorder.component.data_transformation import DataTransformation
from backorder.component.data_validation import DataValidation
from backorder.component.model_evaluation import ModelEvaluation
from backorder.component.model_pusher import ModelPusher
from backorder.component.model_trainer import ModelTrainer
from backorder.config.configuration import Configuartion
from backorder.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from backorder.exception import BackorderException
from backorder.logger import logging


class Experiment(NamedTuple):
    experiment_id: uuid.UUID
    initialization_timestamp: str
    artifact_time_stamp: str
    running_status: bool
    start_time: datetime | None
    stop_time: datetime | None
    execution_time: datetime | None
    message: str
    experiment_file_path: Path
    accuracy: float
    is_model_accepted: bool


class Pipeline(Thread):
    experiment: Experiment
    experiment_file_path: Path

    def __init__(self, config: Configuartion) -> None:
        try:
            config.training_pipeline_config.artifact_dir.mkdir(exist_ok=True)
            Pipeline.experiment_file_path = (
                config.training_pipeline_config.artifact_dir
                / C.EXPERIMENT_DIR_NAME
                / C.EXPERIMENT_FILE_NAME
            )

            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config()
            )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_validation_config=self.config.get_data_validation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise BackorderException(e, exc_info())

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.config.get_model_trainer_config(),  # provided configuration and  artifact entity
                data_transformation_artifact=data_transformation_artifact,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def start_model_pusher(
        self, model_eval_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            logging.info("Pipeline starting.")

            experiment_id = uuid.uuid4()

            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_time_stamp=self.config.time_stamp,
                running_status=True,
                start_time=datetime.now(),
                stop_time=None,
                execution_time=None,
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=False,
                message="Pipeline has been started.",
                accuracy=-1,
            )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_eval_artifact=model_evaluation_artifact
                )
                logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

            stop_time = datetime.now()
            Pipeline.experiment = Experiment(
                experiment_id=Pipeline.experiment.experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_time_stamp=self.config.time_stamp,
                running_status=False,
                start_time=Pipeline.experiment.start_time,
                stop_time=stop_time,
                execution_time=stop_time - Pipeline.experiment.start_time,  # type: ignore
                message="Pipeline has been completed.",
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=model_evaluation_artifact.is_model_accepted,
                accuracy=model_trainer_artifact.model_accuracy,
            )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update(
                    {
                        "created_time_stamp": [datetime.now()],
                        "experiment_file_path": [
                            os.path.basename(Pipeline.experiment.experiment_file_path)
                        ],
                    }
                )

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path, index=False, header=False, mode="a"
                    )
                else:
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path, mode="w", index=False, header=True
                    )
            else:
                print("First start experiment")
        except Exception as e:
            raise BackorderException(e, exc_info()) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(
                    columns=["experiment_file_path", "initialization_timestamp"], axis=1
                )
            else:
                return pd.DataFrame()
        except Exception as e:
            raise BackorderException(e, exc_info()) from e
