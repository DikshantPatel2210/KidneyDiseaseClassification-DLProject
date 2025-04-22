from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataSplitConfig
from cnnClassifier import logger
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split_data(self):
        try:
            df = pd.read_csv(self.config.base_csv_path)
            label_mapping = {
                "Normal": self.config.Normal,
                "Cyst": self.config.Cyst,
                "Tumor": self.config.Tumor,
                "Stone": self.config.Stone}

            df["label"] = df["label"].map(label_mapping)
            logger.info(f"Applied label mapping: {label_mapping}")
            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.test_size,
                stratify=df["label"],
                random_state=self.config.random_state
            )

            test_df, valid_df = train_test_split(
                temp_df,
                test_size=self.config.test_size,
                stratify=temp_df["label"],
                random_state=self.config.random_state
            )

            # Save all
            create_directories([self.config.root_dir])
            train_df.to_csv(self.config.train_csv_path, index=False)
            valid_df.to_csv(self.config.val_csv_path, index=False)
            test_df.to_csv(self.config.test_csv_path, index=False)

            logger.info("Train, validation, and test files saved successfully!")
            logger.info(
                f"""
                Train shape: {train_df.shape}
                Training Set Class Distribution:
                {train_df["label"].value_counts(normalize=True)}

                Validation shape: {valid_df.shape}
                Validation Set Class Distribution:
                {valid_df["label"].value_counts(normalize=True)}

                Test shape: {test_df.shape}
                Test Set Class Distribution:
                {test_df["label"].value_counts(normalize=True)}
                """)

        except Exception as e:
            logger.exception(f"Error in data splitting: {e}")
            raise e