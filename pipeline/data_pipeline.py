from data_loader import load_csv, validate_schema
from features.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from logging_utils.logger import get_logger

import json
from datetime import datetime
from pathlib import Path
import yaml
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent



class DataPipeline:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.logger = get_logger(
            "DataPipeline",
            self.config["logging"]["log_path"]
        )

    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def run(self):
        self.logger.info("Pipeline started")

        # ------------------- Data Loading & Validation -------------------
        file_path = self.config["data"]["file_path"]
        required_columns = self.config["validation"]["required_columns"]

        self.logger.info(f"Loading data from {file_path}")
        df = load_csv(file_path)
        validate_schema(df, required_columns)
        self.logger.info("Schema validation passed")

        # ------------------- Feature Engineering -------------------
        target_column = self.config["training"]["target"]
        feature_engineer = FeatureEngineer(target_column)
        X, y = feature_engineer.fit_transform(df)

        self.logger.info(f"Feature engineering completed | X: {X.shape}, y: {y.shape}")

        # ------------------- Model Training & Evaluation -------------------
        test_size = self.config["training"]["test_size"]
        trainer = ModelTrainer(test_size=test_size)

        X_test, y_test = trainer.train(X, y)
        metrics = trainer.evaluate(X_test, y_test)

        self.logger.info(f"Model evaluation metrics: {metrics}")

        # ------------------- Model Persistence -------------------
        relative_model_path = self.config["training"]["model_path"]
        model_path = PROJECT_ROOT / relative_model_path
        preprocessor_path = PROJECT_ROOT / "artifacts/models/preprocessor.pkl"

        trainer.save_model(model_path)
        joblib.dump(feature_engineer, preprocessor_path)
        self.logger.info(f"Model saved to {model_path} and preprocessor to {preprocessor_path}")

        # ------------------- Inference Sanity Check -------------------
        trainer.load_model(model_path)
        sample_prediction = trainer.model.predict(X_test[:1])
        self.logger.info(f"Sample prediction: {sample_prediction}")

        # ------------------- Run Metadata -------------------
        metadata = {
            "run_time": datetime.utcnow().isoformat(),
            "model_type": self.config["training"]["model_type"],
            "test_size": test_size,
            "metrics": metrics
        }

        metadata_path =PROJECT_ROOT / "artifacts/metadata/run_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Run metadata saved to {metadata_path}")
        self.logger.info("Pipeline completed successfully")
