import logging
import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
import joblib

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_building_step(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    model_path: str = "models/tuned/Xgboost_tuned.pkl"
):
    """
    Loads a tuned XGBClassifier model from disk and optionally retrains it.

    Parameters:
    X_train (pd.DataFrame): Preprocessed training features.
    y_train (pd.Series): Training target labels.
    model_path (str): Path to the saved tuned model.

    Returns:
    model: The loaded (and optionally retrained) XGBClassifier model.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    logging.info(f"Loading tuned model from {model_path}")
    model = joblib.load(model_path)

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Optionally retraining the loaded model on the training data.")
        # If you want to continue training, uncomment next line; otherwise skip
        # model.fit(X_train, y_train)
        logging.info("Model ready for use.")
    except Exception as e:
        logging.error(f"Error during model loading/training: {e}")
        raise e
    finally:
        mlflow.end_run()

    return model
