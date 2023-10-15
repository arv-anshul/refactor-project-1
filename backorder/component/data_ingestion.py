import gzip
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from backorder.entity.artifact_entity import DataIngestionArtifact
from backorder.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
        self.data_ingestion_config = data_ingestion_config

    def download_backorder_data(self) -> Path:
        download_url = self.data_ingestion_config.dataset_download_url
        tgz_download_dir = self.data_ingestion_config.tgz_download_dir

        if tgz_download_dir.exists():
            tgz_download_dir.rmdir()
        tgz_download_dir.mkdir(parents=True, exist_ok=True)

        backorder_file_name = Path(download_url).name
        tgz_file_path = tgz_download_dir / backorder_file_name
        logging.info(f"Downloading file from: {download_url} into: {tgz_file_path}")

        urllib.request.urlretrieve(download_url, tgz_file_path)
        logging.info(f"File: {tgz_file_path} has been downloaded successfully.")
        return tgz_file_path

    def extract_tgz_file(self, tgz_file_path: Path):
        raw_data_dir = self.data_ingestion_config.raw_data_dir

        if raw_data_dir.exists():
            raw_data_dir.unlink()

        raw_data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Extracting tgz file: {tgz_file_path} into dir: {raw_data_dir}")
        raw_file_path = raw_data_dir / tgz_file_path.with_suffix(".gz").name
        if tgz_file_path.exists():
            with gzip.open(tgz_file_path, "rb") as f_in:
                with open(raw_file_path, "wb") as f_out:
                    f_out.write(f_in.read())
        else:
            logging.error(f"The tgz_file_path: {tgz_file_path} does not exist.")

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        raw_data_dir = self.data_ingestion_config.raw_data_dir
        file_list = os.listdir(raw_data_dir)

        file_name = file_list[0]
        backorder_file_path = raw_data_dir / file_name

        logging.info(f"Reading csv file: {backorder_file_path}")
        backorder_data_frame = pd.read_csv(backorder_file_path, low_memory=False)
        backorder_data_frame.drop(
            columns=["Unnamed: 0.1", "Unnamed: 0", "sku"], inplace=True, axis=1
        )

        logging.info("Splitting data into train and test")
        start_train_set = None
        start_test_set = None

        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.4, random_state=42
        )
        for train_index, test_index in stratified_shuffle_split.split(
            backorder_data_frame,
            backorder_data_frame["went_on_backorder"].fillna(
                backorder_data_frame["went_on_backorder"].mode()[0]
            ),
        ):
            start_train_set = backorder_data_frame.loc[train_index]
            start_test_set = backorder_data_frame.loc[test_index]

        train_file_path = self.data_ingestion_config.ingested_train_dir / file_name
        test_file_path = self.data_ingestion_config.ingested_test_dir / file_name

        if start_train_set is not None:
            train_file_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Exporting training dataset to file: {train_file_path}")
            start_train_set.to_csv(train_file_path, index=False)

        if start_test_set is not None:
            test_file_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Exporting test dataset to file: {test_file_path}")
            start_test_set.to_csv(test_file_path, index=False)

        data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=train_file_path,
            test_file_path=test_file_path,
            is_ingested=True,
            message="Data ingestion completed successfully.",
        )
        logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
        return data_ingestion_artifact

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        tgz_file_path = self.download_backorder_data()
        self.extract_tgz_file(tgz_file_path=tgz_file_path)
        return self.split_data_as_train_test()

    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
