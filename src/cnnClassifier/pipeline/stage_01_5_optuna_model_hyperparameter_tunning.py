from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_loader import DataLoader
from cnnClassifier.components.callback import CallbackHandler
from cnnClassifier.components.optuna_model_hyperparameter_tuning_mlflow import OptunaModelTunner
import optuna
from cnnClassifier import logger

STAGE_NAME = "Optuna Hyperparameter Tuning Stage"
class OptunaHyperparameterTuningPipeline:
        def __init__(self):
            pass
        def main(self):
            config = ConfigurationManager()

            # Load data
            data_loader_config = config.get_data_loader_config()
            data_loader = DataLoader(config=data_loader_config)
            train_generator, val_generator, test_generator, train_df, ori_train = data_loader.get_generators()

            # Callbacks
            callbacks_config = config.get_callbacks_config()
            handler = CallbackHandler(config=callbacks_config, ori_training_set=ori_train)
            class_weights = handler.get_class_weights()
            callbacks = handler.get_callbacks()

            # Optuna Config
            optuna_config = config.get_Optuna_config()

            # Optuna Tuner
            optuna_tunner = OptunaModelTunner(
                params=optuna_config,
                training_set=train_generator,
                validation_set=val_generator,
                train_df=train_df,
                callbacks=callbacks,
                class_weights_dict=class_weights,
                class_names=list(ori_train.class_indices.keys())
            )

            # Run Optuna Study
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(),
                                        study_name="kidney_optuna_study",
                                        storage="sqlite:///optuna_study.db",  # this saves to file
                                        load_if_exists=True )
            study.optimize(optuna_tunner.objective, n_trials=30)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = OptunaHyperparameterTuningPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e