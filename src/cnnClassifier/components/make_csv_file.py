import os
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import MakeDataCSVConfig
import pandas as pd

class make_data_csvformat:

    def __init__(self, config: MakeDataCSVConfig):
        self.config = config

    def convert_csv(self):
        try:
            classes = os.listdir(self.config.source_file_path)
            data = []
            for label in classes:
                folder_path = os.path.join(self.config.source_file_path, label)
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    data.append((file_path, label))
            self.df = pd.DataFrame(data, columns=['filepath', 'label'])
            logger.info(f"DataFrame successfully created with shape: {self.df.shape}")
        except Exception as e:
            raise Exception(f"Error in making CSV file: {e}")

    def save_dataframe_to_csv(self, index: bool = False):
        try:
            filepath = Path(self.config.base_csv_file)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(filepath, index=index)
            logger.info(f"CSV file saved at: {filepath}")
        except Exception as e:
            raise Exception(f"Error in downloading file: {e}")
