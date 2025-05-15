from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_loader import DataLoader
from cnnClassifier.components.callback import CallbackHandler
from cnnClassifier.components.optuna_model_testing import OptunaEvaluation
import optuna
from cnnClassifier import logger

STAGE_NAME = "Optuna Model Testing Stage"


class OptunaModelTestingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()

        # Load test generator
        data_loader_config = config_manager.get_data_loader_config()
        data_loader = DataLoader(data_loader_config)
        _, _, test_generator, _, _ = data_loader.get_generators()

        # Get optuna test configuration
        optuna_test_config = config_manager.get_optuna_test_config()

        # Initialize evaluator and run evaluation + logging
        evaluator = OptunaEvaluation(optuna_test_config, test_generator)
        evaluator.evaluate()
        evaluator.log_into_mlflow()


if __name__ == "__main__":

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = OptunaModelTestingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:

        logger.exception(e)

        raise e