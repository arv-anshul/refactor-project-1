from pathlib import Path
from typing import NamedTuple


class DataIngestionArtifact(NamedTuple):
    train_file_path: Path
    test_file_path: Path
    is_ingested: bool
    message: str


class DataValidationArtifact(NamedTuple):
    schema_file_path: Path
    report_file_path: Path
    report_page_file_path: Path
    is_validated: bool
    message: str


class DataTransformationArtifact(NamedTuple):
    is_transformed: bool
    message: str
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    preprocessed_object_file_path: Path


class ModelTrainerArtifact(NamedTuple):
    is_trained: bool
    message: str
    trained_model_file_path: Path
    test_f1_score: float
    train_f1_score: float
    train_accuracy: float
    test_accuracy: float
    model_accuracy: float


class ModelEvaluationArtifact(NamedTuple):
    is_model_accepted: bool
    evaluated_model_path: Path


class ModelPusherArtifact(NamedTuple):
    is_model_pusher: bool
    export_model_file_path: Path
