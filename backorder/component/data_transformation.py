import os

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backorder import constant as C
from backorder.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from backorder.entity.config_entity import DataTransformationConfig
from backorder.io import load_data, read_yaml_file, save_numpy_array_data, save_object
from backorder.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ):
        logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact

    def get_data_transformer_object(self) -> ColumnTransformer:
        schema_file_path = self.data_validation_artifact.schema_file_path
        dataset_schema = read_yaml_file(file_path=schema_file_path)

        numerical_columns = dataset_schema[C.NUMERICAL_COLUMN_KEY]
        categorical_columns = dataset_schema[C.CATEGORICAL_COLUMN_KEY]

        num_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler()),
            ]
        )

        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Numerical columns: {numerical_columns}")

        preprocessing = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ]
        )
        return preprocessing

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Obtaining preprocessing object.")
        preprocessing_obj = self.get_data_transformer_object()

        logging.info("Obtaining training and test file path.")
        train_file_path = self.data_ingestion_artifact.train_file_path
        test_file_path = self.data_ingestion_artifact.test_file_path

        schema_file_path = self.data_validation_artifact.schema_file_path

        logging.info("Loading training and test data as pandas dataframe.")
        train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
        test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

        schema = read_yaml_file(file_path=schema_file_path)
        target_column_name = schema[C.TARGET_COLUMN_KEY]

        logging.info("Splitting input and target feature from training and testing dataframe.")
        input_feature_train_df = train_df.drop(target_column_name, axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(target_column_name, axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying preprocessing object on training dataframe and testing dataframe")
        input_feature_train_arr = preprocessing_obj.fit_transform(
            input_feature_train_df
        )  # column transformation not applied to target of both test and train dataset
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        # Using imblearn to balance the dataset
        over = SMOTE(sampling_strategy=0.8)
        under = RandomUnderSampler(sampling_strategy=0.8)
        steps = [("o", over), ("u", under)]
        pipeline = ImblearnPipeline(steps=steps)
        X_train_resampled, y_train_resampled = pipeline.fit_resample(
            input_feature_train_arr, target_feature_train_df
        )

        X_test_resampled, y_test_resampled = pipeline.fit_resample(
            input_feature_test_arr, target_feature_test_df
        )

        # Concatenate input features and target features
        train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
        test_arr = np.c_[X_test_resampled, np.array(y_test_resampled)]

        transformed_train_dir = self.data_transformation_config.transformed_train_dir
        transformed_test_dir = self.data_transformation_config.transformed_test_dir

        train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
        test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")

        transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
        transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

        logging.info("Saving transformed training and testing array.")
        save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
        save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

        preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

        logging.info("Saving preprocessing object.")
        save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

        data_transformation_artifact = DataTransformationArtifact(
            is_transformed=True,
            message="Data transformation successful.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path,
        )

        logging.info(f"Data transformation artifact: {data_transformation_artifact}")
        return data_transformation_artifact

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
