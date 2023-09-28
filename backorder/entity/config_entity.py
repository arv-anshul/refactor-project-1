from typing import NamedTuple


class DataIngestionConfig(NamedTuple):
    dataset_download_url: str
    tgz_download_dir: str
    raw_data_dir: str
    ingested_train_dir: str
    ingested_test_dir: str


class DataValidationConfig(NamedTuple):
    schema_file_path: str
    report_file_path: str
    report_page_file_path: str


class DataTransformationConfig(NamedTuple):
    transformed_train_dir: str
    transformed_test_dir: str
    preprocessed_object_file_path: str


class ModelTrainerConfig(NamedTuple):
    trained_model_file_path: str
    base_accuracy: float
    model_config_file_path: str


class ModelEvaluationConfig(NamedTuple):
    model_evaluation_file_path: str
    time_stamp: str


class ModelPusherConfig(NamedTuple):
    export_dir_path: str


class TrainingPipelineConfig(NamedTuple):
    artifact_dir: str
