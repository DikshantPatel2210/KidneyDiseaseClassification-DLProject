import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import(
Conv2D, MaxPooling2D, Dense, Dropout,BatchNormalization, GlobalAveragePooling2D)

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import optuna

import mlflow
import mlflow.tensorflow

from cnnClassifier import logger
from mlflow.models import infer_signature
import dagshub

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
from tensorflow.keras import backend as K
from cnnClassifier.entity.config_entity import OptunaConfig

class OptunaModelTunner:
    def __init__(self, params: OptunaConfig, training_set, validation_set, train_df, callbacks, class_weights_dict,
                 class_names):
        self.params = params
        self.training_set = training_set
        self.validation_set = validation_set
        self.train_df = train_df
        self.callbacks = callbacks
        self.class_weights_dict = class_weights_dict
        self.class_names = class_names

    def create_model(self, trial):
        n_conv_layers = trial.suggest_int("n_conv_layers", self.params.min_n_conv_layers, self.params.max_n_conv_layers)
        n_dense_layers = trial.suggest_int("n_dense_layers", self.params.min_n_dense_layers,
                                           self.params.max_n_dense_layers)
        optimizer_name = trial.suggest_categorical('optimizer', self.params.optimizer)
        Conv2D_strides_size = trial.suggest_categorical("Conv2D_strides_size", self.params.Conv2D_strides_size)
        Conv2D_stride = tuple(map(int, Conv2D_strides_size.lower().split("x")))

        Conv2D_kernel_size_str = trial.suggest_categorical("Conv2D_kernel_size", self.params.Conv2D_kernel_size)
        Conv2D_kernel_size = tuple(map(int, Conv2D_kernel_size_str.lower().split("x")))

        MaxPooling2D_strides_size = trial.suggest_categorical("MaxPooling2D_strides_size",
                                                              self.params.MaxPooling2D_strides_size)
        MaxPooling2D_stride = tuple(map(int, MaxPooling2D_strides_size.lower().split("x")))

        MaxPooling2D_kernel_size_str = trial.suggest_categorical("MaxPooling2D_kernel_size",
                                                                 self.params.MaxPooling2D_kernel_size)
        MaxPooling2D_kernel_size = tuple(map(int, MaxPooling2D_kernel_size_str.lower().split("x")))

        lr = trial.suggest_float("learning_rate", 0.00001, 0.001, log=True)
        filter_0 = 32
        model = Sequential()
        model.add(Conv2D(filter_0, Conv2D_kernel_size, strides=Conv2D_stride, activation="relu", padding='same',
                         input_shape=(224, 224, 1), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(MaxPooling2D_kernel_size, strides=MaxPooling2D_stride, padding='same'))

        # for i in range(n_conv_layers):
        #     filters = trial.suggest_categorical(f"filters{i}", self.params.filters)
        #     model.add(Conv2D(filters, Conv2D_kernel_size, strides=Conv2D_stride, activation="relu", padding='same'))
        #     model.add(BatchNormalization())
        #     model.add(MaxPooling2D(MaxPooling2D_kernel_size, strides=MaxPooling2D_stride, padding='same'))
        #
        # model.add(GlobalAveragePooling2D())
        #
        # for i in range(n_dense_layers):
        #     dense_units = trial.suggest_categorical(f"dense_units{i}", self.params.dense_units)
        #     model.add(Dense(dense_units, activation='relu'))
        #     model.add(BatchNormalization())
        #     model.add(Dropout(0.3))
        #
        # model.add(Dense(4, activation='softmax'))
        #
        Allfilters = []

        # Add filter_0 manually if needed
        filter_0 = 32
        Allfilters.append(filter_0)

        if n_conv_layers == 6:
           filter6_1 = trial.suggest_categorical("filter6_1", [32, 64])
           filter6_2= 64
           filter6_3= trial.suggest_categorical("filter6_3", [64, 128])
           filter6_4= 128
           filter6_5 = 512
           Allfilters += [filter6_1, filter6_2, filter6_3, filter6_4, filter6_5]

        elif n_conv_layers == 5:
           filter5_1 = trial.suggest_categorical("filter5_1", [32, 64])
           filter5_2 = trial.suggest_categorical("filter5_2", [64, 128])
           filter5_3 = trial.suggest_categorical("filter5_3", [128, 512])
           filter5_4 = 512
           Allfilters += [filter5_1, filter5_2, filter5_3, filter5_4]

        elif n_conv_layers == 4:
          filter4_1 = trial.suggest_categorical("filter4_1", [32, 64])
          filter4_2 = 64
          filter4_3 = trial.suggest_categorical("filter4_3", [128, 512])
          Allfilters += [filter4_1, filter4_2, filter4_3]

        else:
          filter3_1 = trial.suggest_categorical("filter3_1", [32, 64])
          filter3_2 = trial.suggest_categorical("filter3_2", [64, 128])
          Allfilters += [filter3_1, filter3_2]


        model = Sequential()


         # First layer with input shape
        model.add(Conv2D(Allfilters[0],Conv2D_kernel_size, strides=Conv2D_stride, activation='relu', padding='same', input_shape=(224, 224, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(MaxPooling2D_kernel_size, strides=MaxPooling2D_stride, padding='same'))

        # Remaining conv layers
        for i in range(1, n_conv_layers):
             model.add(Conv2D(Allfilters[i],Conv2D_kernel_size, strides=Conv2D_stride, activation='relu', padding='same'))
             model.add(BatchNormalization())
             model.add(MaxPooling2D(MaxPooling2D_kernel_size, strides=MaxPooling2D_stride, padding='same'))

        model.add(GlobalAveragePooling2D())



        Alldenseunits = []

        if n_dense_layers == 5:
           dense_units5_1 = trial.suggest_categorical("dense_units5_1", [256, 128])
           dense_units5_2 = trial.suggest_categorical("dense_units5_2", [128, 64])
           dense_units5_3 = trial.suggest_categorical("dense_units5_3", [64, 32])
           dense_units5_4 = trial.suggest_categorical("dense_units5_4", [32, 10])
           Alldenseunits = [dense_units5_1, dense_units5_2, dense_units5_3, dense_units5_4]

        elif n_dense_layers == 4:
           dense_units4_1 = trial.suggest_categorical("dense_units4_1", [256, 128])
           dense_units4_2 = trial.suggest_categorical("dense_units4_2", [128, 64])
           dense_units4_3 = trial.suggest_categorical("dense_units4_3", [64, 32])
           Alldenseunits = [dense_units4_1, dense_units4_2, dense_units4_3]

        elif n_dense_layers == 3:
           dense_units3_1 = trial.suggest_categorical("dense_units3_1", [128, 64])
           dense_units3_2 = trial.suggest_categorical("dense_units3_2", [64, 32])
           Alldenseunits = [dense_units3_1, dense_units3_2]

        else:
           dense_units2_1 = trial.suggest_categorical("dense_units2_1", [32, 64])
           Alldenseunits = [dense_units2_1]

        for units in Alldenseunits:
           model.add(Dense(units, activation='relu'))
           model.add(BatchNormalization())
           model.add(Dropout(0.3))

        model.add(Dense(4, activation='softmax'))

        #Optimizer
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, clipvalue=1.0)
        else:
            raise ValueError("Choose 'adam' ,'sgd' or 'RMSprop' for optimizer_name")

        model.compile(optimizer=optimizer,
                      loss=self.params.loss,
                      metrics=self.params.metrics)

        return model

    def objective(self, trial):
        dagshub.init(repo_owner='DikshantPatel2210', repo_name='KidneyDiseaseClassification-DLProject', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/DikshantPatel2210/KidneyDiseaseClassification-DLProject.mlflow")
        mlflow.set_experiment("CNN_Optuna_MLFLOW_Run")
        try:
            if mlflow.active_run():
                mlflow.end_run()

            input_example = np.random.rand(32, 224, 224, 1).astype(float)
            output_example = np.random.rand(32, 4).astype(float)

            with mlflow.start_run(run_name=f"trial_{trial.number}"):
                n_conv_layers = trial.suggest_int("n_conv_layers", 3, 6)
                n_dense_layers = trial.suggest_int("n_dense_layers", 2, 5)
                model = self.create_model(trial)

                # Log hyperparameters
                for param_name, param_value in trial.params.items():
                    mlflow.log_param(param_name, param_value)
                    lower_name = param_name.lower()
                if any(keyword in lower_name for keyword in ["batch", "learning", "optimizer"]):
                    prefix = "training"
                elif any(keyword in lower_name for keyword in ["filter", "kernel", "stride", "conv", "dense", "layer"]):
                    prefix = "model"
                elif any(keyword in lower_name for keyword in ["image", "input"]):
                    prefix = "input"
                elif any(keyword in lower_name for keyword in ["loss"]):
                    prefix = "meta"
                else:
                    prefix = "misc"

                    # Log with prefix
                mlflow.log_param(f"{prefix}/{param_name}", param_value)

                history = model.fit(
                    self.training_set,
                    #steps_per_epoch=5,
                    steps_per_epoch=len(self.train_df) // 32,
                    validation_data=self.validation_set,
                    epochs=self.params.epochs,
                    callbacks=self.callbacks,
                    class_weight=self.class_weights_dict,
                    verbose=1
                )

                # Static params
                mlflow.log_param("input/image_size", "224x224x1")
                mlflow.log_param("model/last_dense_units", 4)
                mlflow.log_param("training/batch_size", 32)
                mlflow.log_param("meta/loss_function", "categorical_crossentropy")

                if n_conv_layers == 6:
                    mlflow.log_param("model/filter6_2", 64)
                    mlflow.log_param("model/filter6_4", 128)
                    mlflow.log_param("model/filter6_5", 512)
                if n_conv_layers == 5:
                    mlflow.log_param("model/filter5_4", 512)
                if n_conv_layers == 4:
                    mlflow.log_param("model/filter4_2", 64)

                #mlflow.log_param(f"dense_units{n_dense_layers}_{n_dense_layers}", 4)

                if np.any(np.isnan(history.history['loss'])) or np.any(np.isnan(history.history['val_loss'])):
                    raise ValueError("NaN value encountered in loss or validation loss.")

                # Log the metrics (train and validation accuracy, loss)
                train_accuracy = max(history.history['accuracy'])  # or 'acc', depending on your Keras version
                train_loss = min(history.history['loss'])
                val_accuracy = max(history.history['val_accuracy'])
                val_loss = min(history.history['val_loss'])

                loss_train = history.history['loss'][-1]
                loss_val = history.history['val_loss'][-1]
                acc_val = history.history['val_accuracy'][-1]
                loss_diff = abs(loss_train - loss_val)
                objective_value = acc_val - loss_diff
                # Log metrics to MLflow
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("train_loss", train_loss)
                mlflow.log_metric("val_accuracy", val_accuracy)
                mlflow.log_metric("val_loss", val_loss)
                mlflow.log_metric("acc_val - loss_diff", objective_value)

                # Log model
                signature = infer_signature(input_example, output_example)
                mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)

                return objective_value

        except Exception as e:
            print(f"[Trial Failed] Error: {e}")
            mlflow.log_param("failed_trial", True)
            mlflow.log_param("error_msg", str(e)[:500])
            return float("nan")

        finally:
            mlflow.end_run()
            try:
                del model
            except:
                pass
            K.clear_session()
            gc.collect()





