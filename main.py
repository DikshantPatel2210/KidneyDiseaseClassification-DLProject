from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
from cnnClassifier.pipeline.stage_01_1_make_csv_file import MakeCSVFilePipeline
from cnnClassifier.pipeline.stage_01_2_data_split import DataSplitPipeline
from cnnClassifier.pipeline.stage_01_3_data_loader import DataLoaderPipeline
from cnnClassifier.pipeline.stage_01_4_callback import CallbackPipeline
#STAGE_NAME = "Data Ingestion stage"
#try:
#  logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#  data_ingestion = DataIngestionTrainingPipeline()
#  data_ingestion.main()
#  logger.info(f">>>> stage ({STAGE_NAME} completed <<<<<<\n\nx============x")
#except Exception as e:
# logger.exception(e)
#  raise e

#STAGE_NAME = "Make_csv_file_ stage"
#try:
#  logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#  make_csv_file = MakeCSVFilePipeline()
#  make_csv_file.main()
#  logger.info(f">>>> stage ({STAGE_NAME} completed <<<<<<\n\nx============x")
#except Exception as e:
#  logger.exception(e)
#  raise e

#STAGE_NAME = "Data Split Stage"

#try:
#   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#  Splitter = DataSplitPipeline()
#  Splitter.main()
#  logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#except Exception as e:
#  logger.exception(e)
#  raise e

#STAGE_NAME = "Data Load Stage"

#try:
#   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#  DataLoader = DataLoaderPipeline()
#  DataLoader.main()
#  logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#except Exception as e:
#  logger.exception(e)
#  raise e

STAGE_NAME = "Callback Stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   Callback = CallbackPipeline()
   Callback.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


#STAGE_NAME = "Prepare base model"

#try:
#  logger.info(f"*********************")
#  logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
#  prepare_base_model = PrepareBaseModelTrainingPipeline()
#  prepare_base_model.main()
#  logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx =========x")
#except Exception as e:
#  logger.exception(e)
#  raise e


#STAGE_NAME = "Training"
#try:
#  logger.info(f"***********************")
#  Training = ModelTrainingPipeline()
#  Training.main()
#  logger.info(f">>>>>>> stage {STAGE_NAME} completed  <<<<<<<<\n\nx=========x")
#except Exception as e:
#  raise e


#STAGE_NAME = "Evalution Stage"
#try:
#   logger.info(f"************************")
#   logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
#   obj = EvaluationPipeline()
#   obj.main()
#   logger.info(f">>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx===============x")
#except Exception as e:
#   logger.exception(e)
#   raise e
