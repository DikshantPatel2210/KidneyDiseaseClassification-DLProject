import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from URL
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            gdown.download(url=dataset_url, output=zip_download_dir, quiet=False, fuzzy=True)

            logger.info(f"Downloaded data successfully to {zip_download_dir}")

        except Exception as e:
            raise Exception(f"Error in downloading file: {e}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        """
        try:
            unzip_path = self.config.unzip_dir
            zip_path = self.config.local_data_file

            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ZIP file not found at {zip_path}")

            if not zipfile.is_zipfile(zip_path):
                raise ValueError(f"The file is not a valid ZIP file: {zip_path}")

            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f" Successfully extracted files to {unzip_path}")

        except Exception as e:
            raise Exception(f"Error in extracting zip file: {e}")