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

