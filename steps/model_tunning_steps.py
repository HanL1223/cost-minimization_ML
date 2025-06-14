import logging
import yaml
import os
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from joblib import dump
from src.tune_model import ModelTunning

def model_tunning_steps(best_model_name,best_model,X_train,y_train,cv,n_iter,random_state = 42):
    config_path = 'config'
    tuner = ModelTunning(config_dir=config_path)
    tuned_model = tuner.tune(best_model_name, best_model, X_train, y_train, cv=5, n_iter=30, random_state=42)
    model_path = tuner.save_tuned_model(best_model_name,tuned_model)
    return model_path