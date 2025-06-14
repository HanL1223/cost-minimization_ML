from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import logging
from typing import Annotated
from zenml import step

import mlflow
import pandas as pd
from src.model_selection import CrossValidationEvaluator,ModelEvaluator

# Get the active experiment tracker from ZenML

@step
def model_selection_steps(X_train,y_train):
    models = []  # Empty list to store all the models

    # Appending models into the list

    models.append(
        ("Logistic Regression", LogisticRegression(solver="newton-cg", random_state=1))
    )
    models.append(("dtree", DecisionTreeClassifier(random_state=1)))
    models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))
    
    strategy = CrossValidationEvaluator(n_split = 5 ,random_state=42)
    evaluator = ModelEvaluator(strategy)

    results  = evaluator.evaluate_models(models,X_train,y_train)
    best_model_name, best_model = evaluator.get_best_model(models)
    model_path = ModelEvaluator.save_best_model(best_model_name, best_model)
    return best_model_name,best_model,model_path

