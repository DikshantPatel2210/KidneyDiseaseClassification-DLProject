from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_loader import DataLoader
from cnnClassifier import logger

STAGE_NAME = "Data Load Stage"

class DataLoaderPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_loader_config = config.get_data_loader_config()
        data_loader = DataLoader(config=data_loader_config)
        train_generator, valid_generator, test_generator,train_df, ori_train = data_loader.get_generators()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataLoaderPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e