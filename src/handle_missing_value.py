import logging
from abc import ABC, abstractmethod

import pandas as pd

# Set up logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Dropping Missing Value
class MissingValueHandling(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for handling missing values in a DataFrame.
        """
        pass


class DropMissingValue(MissingValueHandling):
    def __init__(self, axis=0, thresh=None):
        """
        Initialized the DropMissingValue with parameters

        Parameters:
        axis: int 0 to drop rows with missing value and 1 drop columns with missing values
        thresh: int The threshold for non-NA value, row/columns with less than thresh non NA value will be dropped
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping missing value with axis = {self.axis}, thresh = {self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing Value dropped")
        return df_cleaned


# Filling Missing Value
class FillMissingValue(MissingValueHandling):  # Should inherit from MissingValueHandling
    def __init__(self, method='mean', fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:  # Fixed typo: DataFrmae -> DataFrame
        logging.info(f"Filling missing values using method {self.method}")
        df_clean = df.copy()
        if self.method == 'mean':
            numeric_cols = df_clean.select_dtypes(include='number').columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df[numeric_cols].mean())
        elif self.method == 'median':
            numeric_cols = df_clean.select_dtypes(include='number').columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df[numeric_cols].median())
        elif self.method == 'mode':
            for column in df_clean.columns:
                df_clean[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_clean = df_clean.fillna(self.fill_value)  # Fixed variable name: df_cleaned -> df_clean
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")
        logging.info("Missing values filled")
        return df_clean


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandling):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandling): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandling):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandling): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


if __name__ == "__main__":
    # Example dataframe
    data = {'A': [1, 2, None, 4], 'B': [5, None, None, 8], 'C': [9, 10, 11, 12]}
    df = pd.DataFrame(data)

    # Initialize missing value handler with drop strategy
    missing_value_handler = MissingValueHandler(DropMissingValue(axis=0, thresh=2))
    df_cleaned = missing_value_handler.handle_missing_values(df)
    print("After dropping:")
    print(df_cleaned)

    # Switch to filling missing values with mean
    missing_value_handler.set_strategy(FillMissingValue(method='mean'))
    df_filled = missing_value_handler.handle_missing_values(df)
    print("\nAfter filling:")
    print(df_filled)