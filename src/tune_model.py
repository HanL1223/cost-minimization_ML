import logging
import yaml
import os
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from joblib import dump
from abc import ABC,abstractmethod
logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')

#Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#Scorer
def minimum_vs_model_cost(y_true, y_pred, cost_params={'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    min_cost = (TP + FN) * cost_params['fn_cost']
    model_cost = (TP * cost_params['tp_cost'] + FP * cost_params['fp_cost'] + FN * cost_params['fn_cost'])
    return min_cost / model_cost

custom_scorer = make_scorer(minimum_vs_model_cost, greater_is_better=True)


class ModelTunning:
    def __init__(self,config_dir ='config'):
        self.config_dir = config_dir

    def load_param_grid(self,model_name:str) -> dict:
        config_path = os.path.join(self.config_dir,f"{model_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file for {model_name} not found at {config_path}")
        with open (config_path,'r') as f:
            param_grid = yaml.safe_load(f)

        logging.info(f"Loaded tuning config for {model_name}, from {config_path}")
        return param_grid
    def tune(self, best_model_name: str, best_model, X, y, cv=3, n_iter=20, random_state=1):
        param_grid = self.load_param_grid(best_model_name)

        randomized_cv = RandomizedSearchCV(
            estimator=best_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=custom_scorer,
            cv=cv,
            random_state=random_state,
            n_jobs=1
        )
        logging.info(f"Starting hyperparameter tuning for {best_model_name} ...")
        randomized_cv.fit(X, y)
        logging.info(f"Best parameters for {best_model_name}: {randomized_cv.best_params_}")
        logging.info(f"Best CV score: {randomized_cv.best_score_:.4f}")
        return randomized_cv.best_estimator_
    @staticmethod
    def save_tuned_model(name: str, model, path="models/tuned"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{name}_tuned.pkl")
        dump(model, model_path)
        logging.info(f"Tuned model '{name}' saved to: {model_path}")
        return model_path
if __name__ == "__main__":
    pass
