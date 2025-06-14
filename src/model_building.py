import logging
from abc import ABC, abstractmethod
from typing import Any
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """
        Abstract method to build and train a model.
        """
        pass


# Concrete Strategy: Use a pre-tuned XGBClassifier
class XGBClassifierStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Loads a tuned XGBClassifier model and fits it on the provided training data.
        """
        model_path = "models/tuned/Xgboost_tuned.pkl"

        logging.info(f"Loading tuned XGBClassifier from {model_path}")
        model = joblib.load(model_path)

        if not isinstance(model, XGBClassifier):
            raise TypeError("Loaded model is not an instance of XGBClassifier.")

        logging.info("Training the loaded XGBClassifier on the provided training data.")
        model.fit(X_train, y_train)

        logging.info("Model training complete.")
        return model


# Context class
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        return self._strategy.build_and_train_model(X_train, y_train)


# Example usage
if __name__ == "__main__":
    # Replace this with your actual data loading
    # df = pd.read_csv("your_dataset.csv")
    # X_train = df.drop(columns=["target"])
    # y_train = df["target"]

    # For demonstration, dummy data
    X_train = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    y_train = pd.Series([0, 1, 0])

    builder = ModelBuilder(XGBClassifierStrategy())
    model = builder.build_model(X_train, y_train)

    # Use model.predict(X_test) on your real test set as needed
    # y_pred = model.predict(X_test)
