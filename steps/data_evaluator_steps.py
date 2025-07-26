import logging
from typing import Tuple, Dict

import pandas as pd
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step

@step(enable_cache=False)
def model_evaluator_step(
    trained_model,  # This is the loaded model, NOT a pipeline
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, float], float]:
    """
    Evaluates the trained classification model using ModelEvaluator and ClassificationModelEvaluationStrategy.

    Parameters:
    trained_model: Loaded trained model (e.g., XGBClassifier instance)
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): True labels for test data.

    Returns:
    Tuple containing:
    - dict: Dictionary of evaluation metrics.
    - float: The custom "Minimum_Vs_Model_cost" metric for cost comparison.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Evaluating the trained classification model.")

    # Initialize evaluator with your classification strategy
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())

    # Evaluate model
    evaluation_metrics = evaluator.evaluate(trained_model, X_test, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")

    min_vs_model_cost = evaluation_metrics.get("Minimum_Vs_Model_cost", None)

    return evaluation_metrics, min_vs_model_cost
