from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_loader import DataLoader
from cnnClassifier.components.callback import CallbackHandler
from cnnClassifier import logger
from cnnClassifier.components.callback import CallbacksConfig
STAGE_NAME = "Callback Stage"

class CallbackPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        data_loader_config = config.get_data_loader_config()
        data_loader = DataLoader(config=data_loader_config)
        train_generator, val_generator, test_generator, ori_train = data_loader.get_generators()

        callbacks_config = config.get_callbacks_config()
        handler = CallbackHandler(config=callbacks_config, ori_training_set=ori_train)
        class_weights = handler.get_class_weights()
        callbacks = handler.get_callbacks()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CallbackPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e