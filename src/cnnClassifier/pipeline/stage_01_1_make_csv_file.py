
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.make_csv_file import make_data_csvformat
from cnnClassifier import logger

STAGE_NAME = "Make_csv_file stage"
class MakeCSVFilePipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        make_data_csv_config = config.get_make_data_csv_config()
        make_data_CSVformat = make_data_csvformat(config=make_data_csv_config)
        make_data_CSVformat.convert_csv()
        make_data_CSVformat.save_dataframe_to_csv()

if __name__ =='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = MakeCSVFilePipeline()
        obj.main()
        logger.info(f">>>> stage ({STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)


