from pathlib import Path
from typing import NamedTuple


class DataIngestionConfig(NamedTuple):
    dataset_download_url: str
    tgz_download_dir: Path
    raw_data_dir: Path
    ingested_train_dir: Path
    ingested_test_dir: Path


class DataValidationConfig(NamedTuple):
    schema_file_path: Path
    report_file_path: Path
    report_page_file_path: Path


class DataTransformationConfig(NamedTuple):
    transformed_train_dir: Path
    transformed_test_dir: Path
    preprocessed_object_file_path: Path


class ModelTrainerConfig(NamedTuple):
    trained_model_file_path: Path
    base_accuracy: float
    model_config_file_path: Path


class ModelEvaluationConfig(NamedTuple):
    model_evaluation_file_path: Path
    time_stamp: str


class ModelPusherConfig(NamedTuple):
    export_dir_path: Path


class TrainingPipelineConfig(NamedTuple):
    artifact_dir: Path
