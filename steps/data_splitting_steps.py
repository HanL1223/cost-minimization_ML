from sklearn.model_selection import train_test_split
import pandas as pd
from zenml import step
from src.data_splitter import DataSplitter,SimpleTrainTestSplitStrategy

@step
def data_spilting_step(df:pd.DataFrame,target_column:str = 'Target'):
    splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size =0.2))
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test