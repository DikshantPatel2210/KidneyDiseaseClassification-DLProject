from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir :Path

@dataclass(frozen = True)
class MakeDataCSVConfig:
    root_dir: Path
    base_csv_file: Path
    source_file_path: Path

@dataclass(frozen = True)
class DataSplitConfig:
    root_dir: Path
    base_csv_path: Path
    train_csv_path: Path
    val_csv_path: Path
    test_csv_path: Path
    test_size: float
    random_state: float
    Normal: int
    Cyst: int
    Tumor:int
    Stone:int

@dataclass
class DataLoaderConfig:
    root_dir: str
    train_data: str
    valid_data: str
    test_data: str
    target_size: tuple
    batch_size: int
    color_mode: str
    class_mode: str
    seed: int
    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    shear_range: float
    zoom_range: float
    horizontal_flip: bool
    fill_mode: str

@dataclass
class CallbacksConfig:
    checkpoint_path: str
    early_stopping_params: dict
    reduce_lr_params: dict
    checkpoint_params: dict

@dataclass(frozen = True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen = True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int