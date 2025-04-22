from cnnClassifier import logger
from cnnClassifier.entity.config_entity import CallbacksConfig
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback
)
from sklearn.utils.class_weight import compute_class_weight
from cnnClassifier import logger


class CustomObjectiveLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc_val = logs.get('val_accuracy', 0)
        acc_train = logs.get('accuracy', 0)
        loss_val = logs.get('val_loss', 0)
        loss_train = logs.get('loss', 0)

        loss_diff = abs(loss_val - loss_train)
        objective_value = acc_val - loss_diff
        logs['val_objective'] = objective_value

        logger.info(
            f"[Epoch {epoch + 1:03d}] Custom Objective = {objective_value:.6f}, "
            f"Train Acc = {acc_train:.4f}, Train Loss = {loss_train:.4f}, "
            f"Val Acc = {acc_val:.4f}, Val Loss = {loss_val:.4f}"
        )


class CallbackHandler:
    def __init__(self, config: CallbacksConfig, ori_training_set):
        self.config = config
        self.ori_training_set = ori_training_set

    def get_class_weights(self):
        logger.info("Computing class weights...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.ori_training_set.classes),
            y=self.ori_training_set.classes
        )
        class_weights_dict = dict(zip(np.unique(self.ori_training_set.classes), class_weights))
        logger.info(f"Class Weights: {class_weights_dict}")
        return class_weights_dict

    def get_callbacks(self):
        logger.info("Preparing callbacks...")

        checkpoint = ModelCheckpoint(
            **self.config.checkpoint_params
        )

        early_stopping = EarlyStopping(
            **self.config.early_stopping_params
        )

        reduce_lr = ReduceLROnPlateau(
            **self.config.reduce_lr_params
        )

        custom_logger = CustomObjectiveLogger()

        return [custom_logger, early_stopping, reduce_lr, checkpoint]