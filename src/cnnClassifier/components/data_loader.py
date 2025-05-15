from cnnClassifier.entity.config_entity import DataLoaderConfig
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnnClassifier import logger
import os


class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def create_generator(self):
        aug = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            shear_range=self.config.shear_range,
            zoom_range=self.config.zoom_range,
            horizontal_flip=self.config.horizontal_flip,
            fill_mode=self.config.fill_mode,
        )
        ori = ImageDataGenerator(rescale=1.0 / 255)
        logger.info("Generators created successfully.")
        return aug, ori

    def load_dataframe(self, file_path):
        logger.info(f"Loading dataframe from file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataframe loaded successfully with {len(df)} records.")
            return df
        except Exception as e:
            logger.error(f"Error loading dataframe from {file_path}: {str(e)}")
            raise e

    def validate_filepaths(self, df):
        # Check if the file paths are valid and log invalid paths
        df['valid_filepath'] = df['filepath'].apply(lambda x: os.path.isfile(x))
        invalid_files = df[df['valid_filepath'] == False]

        if not invalid_files.empty:
            logger.warning(f"Found {len(invalid_files)} invalid file paths!")
            logger.warning(f"Invalid file paths: \n{invalid_files[['filepath']]}")

        logger.info(f"Valid file paths count: {df['valid_filepath'].sum()}")
        # Return only rows with valid file paths
        return df[df['valid_filepath']]

    def get_flow(self, df, generator, shuffle=True):
        return generator.flow_from_dataframe(
            dataframe=df,
            x_col="filepath",
            y_col="label",
            target_size=(224, 224),
            batch_size=self.config.batch_size,
            class_mode=self.config.class_mode,
            color_mode=self.config.color_mode,
            shuffle=shuffle,
            seed=self.config.seed
        )

    def combined_generator(self, aug, ori):
        n_orig = int(0.5 * len(aug))
        n_aug = len(aug) - n_orig
        while True:
            aug_images, aug_labels = next(aug)
            ori_images, ori_labels = next(ori)

            # Combine the two datasets
            images = np.concatenate((ori_images[:n_orig], aug_images[n_aug:]), axis=0)
            labels = np.concatenate((ori_labels[:n_orig], aug_labels[n_aug:]), axis=0)

            yield images, labels

    def get_generators(self):
        # File paths for the CSVs
        train_path = f"{self.config.root_dir}/{self.config.train_data}"
        valid_path = f"{self.config.root_dir}/{self.config.valid_data}"
        test_path = f"{self.config.root_dir}/{self.config.test_data}"

        # Load the dataframes
        try:
            train_df = self.load_dataframe(train_path)
            valid_df = self.load_dataframe(valid_path)
            test_df = self.load_dataframe(test_path)
        except Exception as e:
            logger.error(f"Error loading dataframes: {str(e)}")
            raise e

        # Validate file paths
        train_df = self.validate_filepaths(train_df)
        valid_df = self.validate_filepaths(valid_df)
        test_df = self.validate_filepaths(test_df)

        # Ensure the labels are in the correct format (str)
        train_df['label'] = train_df['label'].astype(str)
        valid_df['label'] = valid_df['label'].astype(str)
        test_df['label'] = test_df['label'].astype(str)

        # Create generators
        aug_gen, ori_gen = self.create_generator()

        # Get augmented and original image generators
        aug_train = self.get_flow(train_df, aug_gen, shuffle=True)
        ori_train = self.get_flow(train_df, ori_gen, shuffle=True)

        # Combine the generators
        train = self.combined_generator(aug_train, ori_train)
        valid = self.get_flow(valid_df, ori_gen, shuffle=False)
        test = self.get_flow(test_df, ori_gen, shuffle=False)
        steps_per_epoch_train = len(train_df) // self.config.batch_size
        logger.info(f"lengh of train_df: {len(train_df)} & Steps per epoch for training: {steps_per_epoch_train}")
        logger.info(f"Successfully created train{train}, validation{valid}, and test{test} generators.")
        return train, valid, test, train_df, ori_train

