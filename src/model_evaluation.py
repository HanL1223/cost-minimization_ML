import logging
from abc import ABC, abstractmethod
import pickle
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix
)
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self,model,X_test,y_test):
        """
        model:the trained model in picket file
        X_test:Testing indenpent variable
        y_test:Testing dependent variable
        """
        pass

class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model, X_test, y_test):
        logging.info("Predictiong using the trained model")
        y_pred = model.predict(X_test)
        logging.info("Calculating classification metrics...")

        TP = confusion_matrix(y_test, y_pred)[1, 1]
        FP = confusion_matrix(y_test, y_pred)[0, 1]
        FN = confusion_matrix(y_test, y_pred)[1, 0]

        Cost = TP * 15 + FP * 5 + FN * 40  # maintenance cost by using model
        Min_Cost = (
            TP + FN
        ) * 15  # minimum possible maintenance cost = number of actual positives
        Percent = (
            Min_Cost / Cost
        )  # ratio of minimum possible maintenance cost and maintenance cost by model

        acc = accuracy_score(y_test, y_pred)  # to compute Accuracy
        recall = recall_score(y_test, y_pred)  # to compute Recall
        precision = precision_score(y_test, y_pred)  # to compute Precision
        f1 = f1_score(y_test, y_pred)  # to compute F1-score
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Minimum_Vs_Model_cost":Percent}
    

if __name__ == "__main__":
    pass
