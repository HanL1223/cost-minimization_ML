import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix
from typing import List, Tuple, Callable
import os
from joblib import dump


#Set up logging
logging.basicConfig(level=logging.INFO,format  = '%(asctime)s - %(levelname)s - %(message)s')


#Abstract base class
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self ,model:list[tuple[str,object]],X:pd.DataFrame,y:pd.Series) ->dict:
        pass

class CrossValidationEvaluator(ModelEvaluationStrategy):
    def __init__(self,n_split = 5, random_state = 1,scorer = None,cost_params = {'tp_cost': 15, 'fp_cost': 5, 'fn_cost': 40}):
        self.n_splits = n_split
        self.random_state = random_state
        self.cost_params = cost_params
        self.scorer = scorer or self._create_default_scorer()

    def _create_default_scorer(self):
        """Creat ethe default Minimum VS model cost scorer"""
        def minimum_vs_model_cost(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            TP = cm[1, 1]
            FP = cm[0, 1]
            FN = cm[1, 0]
            min_cost = (TP + FN) * self.cost_params['fn_cost']
            model_cost = (TP * self.cost_params['tp_cost'] + 
                          FP * self.cost_params['fp_cost'] + 
                          FN * self.cost_params['fn_cost'])
            return min_cost/model_cost
        
        return make_scorer(minimum_vs_model_cost,greater_is_better= True)
    def evaluate(self, models: List[Tuple[str, object]], X: pd.DataFrame, y: pd.Series) -> dict:
        logging.info("Starting CV evaluation with maintenance cost scoring")
        results = {}
        kfold = StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=self.random_state)
        for name,model in models:
            try:
                cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=self.scorer)
                results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_dev': cv_scores.std(),
                    'all_scores': cv_scores,
                    'cost_ratio': cv_scores.mean()  # The score is already a cost ratio
                }
                logging.info(f"{name}: Mean cost ratio = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                results[name] = {
                    'error': str(e),
                    'mean_score': np.nan,
                    'std_dev': np.nan
                }
        return results
    
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy
        self._last_score = None
        
    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """Set a new evaluation strategy"""
        logging.info("Changing evaluation strategy")
        self._strategy = strategy
        
    def evaluate_models(self, models: List[Tuple[str, object]], X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate models using the current strategy"""
        logging.info("Evaluating models with maintenance cost optimization")
        self._last_results = self._strategy.evaluate(models, X, y)
        return self._last_results
    def get_best_model(self, models: List[Tuple[str, object]]) -> Tuple[str, object]:
        if not self._last_results:
            raise ValueError("No evaluation results available. Run `evaluate_models()` first.")

        best_model_name = None
        best_score = float("-inf")
        
        for name, result in self._last_results.items():
            if 'mean_score' in result and result['mean_score'] > best_score:
                best_score = result['mean_score']
                best_model_name = name

        if best_model_name is None:
            raise ValueError("Could not determine the best model. Check evaluation results.")

        # Retrieve the actual model object from input
        model_dict = dict(models)
        return best_model_name, model_dict[best_model_name]
    @staticmethod
    def save_best_model(name: str, model, path="models"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{name}_base.pkl")
        dump(model, model_path)
        logging.info(f"Best model '{name}' saved to: {model_path}")
        return model_path


    # Example usage
    if __name__ == "__main__":
        pass

