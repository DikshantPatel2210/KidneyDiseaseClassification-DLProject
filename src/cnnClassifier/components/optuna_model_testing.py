import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import OptunaTestingConfig
import dagshub
from cnnClassifier import logger

class OptunaEvaluation:
    def __init__(self, config: OptunaTestingConfig, test_generator):
        self.config = config
        self.test_generator = test_generator
        logger.info("Initialized OptunaEvaluation with given configuration.")

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        logger.info(f"Loading model from path: {path}")
        return tf.keras.models.load_model(path)

    def evaluate(self):
        logger.info("Starting model evaluation on test dataset...")
        self.model = self.load_model(Path(self.config.optuna_best_trained_model))
        self.score = self.model.evaluate(self.test_generator, verbose=1)
        logger.info(f"Model evaluation completed. Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self.save_score()

    def save_score(self):
        scores = {"optuna_test_loss": self.score[0], "optuna_test_accuracy": self.score[1]}
        score_path = Path(self.config.optuna_test_scores)
        create_directories([score_path.parent])
        save_json(score_path, data=scores)
        logger.info(f"Saved evaluation scores at: {score_path}")

    def log_into_mlflow(self):
        logger.info("Logging evaluation results into MLflow...")
        dagshub.init(repo_owner='DikshantPatel2210', repo_name='KidneyDiseaseClassification-DLProject', mlflow=True)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Optuna-BestModel-TestEval-v1")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name="Optuna-BestModel-TestEval"):
            mlflow.log_param("model_type", "OptunaBest")
            mlflow.log_metrics({"optuna_test_loss": self.score[0], "optuna_test_accuracy": self.score[1]})
            logger.info("Logged parameters and metrics to MLflow.")

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="OptunaBestModel")
                logger.info("Model registered to MLflow Registry.")
            else:
                mlflow.keras.log_model(self.model, "model")
                logger.info("Model logged to MLflow without registration.")