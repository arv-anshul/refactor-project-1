import os
from datetime import datetime
from pathlib import Path


def get_current_time_stamp() -> str:
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


ROOT_DIR = Path.cwd()
CONFIG_DIR = Path("config")
CONFIG_FILE_NAME = Path("config.yaml")
CONFIG_FILE_PATH = ROOT_DIR / CONFIG_DIR / CONFIG_FILE_NAME
CURRENT_TIME_STAMP = get_current_time_stamp()

# Training pipeline
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = Path("artifact_dir")
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

# Data Ingestion
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = Path("data_ingestion")
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = Path("raw_data_dir")
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = Path("tgz_download_dir")
DATA_INGESTION_INGESTED_DIR_NAME_KEY = Path("ingested_dir")
DATA_INGESTION_TRAIN_DIR_KEY = Path("ingested_train_dir")
DATA_INGESTION_TEST_DIR_KEY = Path("ingested_test_dir")

# Schema validation
SCHEMA_VALIDATION_COLUMNS_KEY = "columns"
SCHEMA_VALIDATION_TARGET_COLUMN_KEY = "target_columns"

# Data Validation
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = Path("schema_file_name")
DATA_VALIDATION_SCHEMA_DIR_KEY = Path("schema_dir")
DATA_VALIDATION_ARTIFACT_DIR_NAME = Path("data_validation")
DATA_VALIDATION_REPORT_FILE_NAME_KEY = Path("report_file_name")
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = Path("report_page_file_name")

# Data Transformation
DATA_TRANSFORMATION_ARTIFACT_DIR = Path("data_transformation")
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY = "add_bedroom_per_room"
DATA_TRANSFORMATION_DIR_NAME_KEY = Path("transformed_dir")
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = Path("transformed_train_dir")
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = Path("transformed_test_dir")
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = Path("preprocessing_dir")
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = Path("preprocessed_object_file_name")

DATASET_SCHEMA_COLUMNS_KEY = "columns"
NUMERICAL_COLUMN_KEY = "numerical_columns"
CATEGORICAL_COLUMN_KEY = "categorical_columns"
TARGET_COLUMN_KEY = "target_columns"

# Model Training
MODEL_TRAINER_ARTIFACT_DIR = Path("model_trainer")
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = Path("trained_model_dir")
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "Path(model_file_name")
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = Path("model_config_dir")
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = Path("model_config_file_name")

# Model Evaluation
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY = Path("model_evaluation_file_name")
MODEL_EVALUATION_ARTIFACT_DIR = Path("model_evaluation")

# Model Pusher
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = Path("model_export_dir")

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = Path("model_path")

EXPERIMENT_DIR_NAME = Path("experiment")
EXPERIMENT_FILE_NAME = Path("experiment.csv")
