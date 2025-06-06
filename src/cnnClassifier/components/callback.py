from cnnClassifier import logger
from cnnClassifier.entity.config_entity import CallbacksConfig
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback,
    TerminateOnNaN,
)
from sklearn.utils.class_weight import compute_class_weight
from cnnClassifier import logger
from datetime import datetime

class CustomObjectiveLogger(Callback):
    def __init__(self):
        super().__init__()
        self.best_objective = float('-inf')
    def on_epoch_end(self, epoch, logs=None):

        acc_val = logs.get('val_accuracy', 0)
        acc_train = logs.get('accuracy', 0)
        loss_val = logs.get('val_loss', 0)
        loss_train = logs.get('loss', 0)


        loss_diff = abs(loss_val - loss_train)
        objective_value = acc_val - loss_diff
        logs['val_objective'] = objective_value

        improved = objective_value > self.best_objective
        if improved:
            self.best_objective = objective_value

        # Add timestamp (optional)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
             f"\n{'='*70}\n"
            f"📅 [{timestamp}] Epoch {epoch + 1:02d}\n"
            f"{'-'*70}\n"
            f"🔹 Train   -> Accuracy: {acc_train:.4f} | Loss: {loss_train:.4f}\n"
            f"🔹 Val     -> Accuracy: {acc_val:.4f} | Loss: {loss_val:.4f}\n"
            f"🎯 Objective (Val Acc - |Loss Diff|): {objective_value:.6f}\n"
            f"{'✅ IMPROVED' if improved else '⚠️  No Improvement'} from {self.best_objective if not improved else objective_value:.6f}\n"
            f"{'='*70}\n"
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
        terminate_on_nan = TerminateOnNaN()
        return [custom_logger, terminate_on_nan, early_stopping, reduce_lr, checkpoint]