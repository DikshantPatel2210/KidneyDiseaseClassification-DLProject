from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig,
                                                TrainingConfig, EvaluationConfig, MakeDataCSVConfig,
                                                DataSplitConfig,DataLoaderConfig, CallbacksConfig )
import os

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_roots])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_make_data_csv_config(self) -> MakeDataCSVConfig:
        config = self.config.make_data_csv

        create_directories([config.root_dir])
        make_data_csv_config = MakeDataCSVConfig(
            root_dir=config.root_dir,
            base_csv_file=config.base_csv_file,
            source_file_path=config.source_file_path,
        )

        return make_data_csv_config

    def get_data_split_config(self) -> DataSplitConfig:
        config = self.config.data_split
        label_mapping = self.params["label_mapping"]
        create_directories([config.root_dir])
        data_split_config = DataSplitConfig(
            root_dir=config.root_dir,
            base_csv_path=config.base_csv_path,
            train_csv_path=config.train_csv_path,
            val_csv_path=config.val_csv_path,
            test_csv_path=config.test_csv_path,
            test_size=config.test_size,
            random_state=config.random_state,
            Normal=label_mapping["Normal"],
            Cyst= label_mapping["Cyst"],
            Tumor=label_mapping["Tumor"],
            Stone=label_mapping["Stone"],
        )
        return data_split_config

    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader
        params = self.params.data_loader
        return DataLoaderConfig(
            root_dir=config.root_dir,
            train_data=config.train_data,
            valid_data=config.valid_data,
            test_data=config.test_data,
            target_size=params.target_size,
            batch_size=params.batch_size,
            color_mode=params.color_mode,
            class_mode=params.class_mode,
            seed=params.seed,
            rotation_range=params.rotation_range,
            width_shift_range=params.width_shift_range,
            height_shift_range=params.height_shift_range,
            shear_range=params.shear_range,
            zoom_range=params.zoom_range,
            horizontal_flip=params.horizontal_flip,
            fill_mode=params.fill_mode
        )

    def get_callbacks_config(self) -> CallbacksConfig:
        callbacks_config = self.config.callbacks
        training_params = self.params.training

        create_directories([callbacks_config.checkpoint_dir])

        return CallbacksConfig(
            checkpoint_path=training_params.checkpoint.filepath,
            early_stopping_params=training_params.early_stopping,
            reduce_lr_params=training_params.reduce_lr,
            checkpoint_params=training_params.checkpoint
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,
                                     "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
            mlflow_uri="https://dagshub.com/DikshantPatel2210/KidneyDiseaseClassification-DLProject.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config