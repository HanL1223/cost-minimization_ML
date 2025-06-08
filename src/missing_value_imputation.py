import logging
from abc import ABC, abstractmethod

import pandas as pd

#set up logging
logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')


#abstract base class

class MissingValueHandling(ABC):
    @abstractmethod
    def handle (self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass



#Concrete Strategy for Dropping Missing Values
class DropMissingValue(MissingValueHandling):
    def __init__(self,axis = 0, thresh=None):
         """
        Initializes the DropMissingValues Strategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
         self.axis = axis
         self.thresh = thresh
    def handle(self,df:pd.DataFrame) -> pd.DataFrame:
         """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
         logging.info(f"Dropping missing value  with axis=  {self.axis} and thresh = {self.thresh}")
         df_cleaned = df.dropna(axis=self.axis,thresh=self.thresh)
         logging.info("Missing values dropped.")
         return df_cleaned
    
class FillMissingValue(MissingValueHandling):
    def __init__(self,method = 'mean',fill_value = None):
        """
    Initializes the FillMissingValuesStrategy with a specific method or fill value.

    Parameters:
    method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
    fill_value (any): The constant value to fill missing values when method='constant'.
    """
        self.method = method
        self.fill_value = fill_value

    def handle(self,df:pd.DataFrame) ->pd.DataFrame:
        """
        
        """
        logging.info(f"Filling missing value with {self.method} strategy")
        df_cleaned = df.copy()
        if self.method == 'mean':
            df_cleaned = df_cleaned.fillna(df_cleaned.select_dtypes('number').mean())
        elif self.method == 'median':
            df_cleaned = df_cleaned.fillna(df_cleaned.select_dtypes('number').median())
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")
        logging.info("Missing values filled.")
        return df_cleaned
    
# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandling):
        self._strategy = strategy

    def set_strategy(self,strategy:MissingValueHandling):
        logging.info("Switching missing value handling strategy.")
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

# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)

    pass

