import logging
from abc import ABC, abstractmethod
import pandas as pd 
from sklearn.model_selection import train_test_split

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_col: str):
        """
        Abstract method to split the data into training and testing sets.
        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.
        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_seed=23):
        self.test_size = test_size
        self.random_seed = random_seed  # Fixed: Changed from random_state to random_seed

    def split_data(self, df, target_col):
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed  # Fixed: Changed to random_seed
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test

# DataSplitter should NOT inherit from DataSplittingStrategy
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.
        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.
        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.
        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)
