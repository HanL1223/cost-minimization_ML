import pandas as pd
from src.missing_value_imputation import *
from zenml import steps

@steps
def missing_value_imputation(df:pd.DataFrame,strategy:str = 'mean') -> pd.DataFrame:
    if strategy =='drop':
        handler = MissingValueHandler(DropMissingValue(axis = 0))
    elif strategy in ['mean','median','constant','mode']:
        handler = MissingValueHandler.set_strategy(FillMissingValue(method =strategy))
    else:
        raise ValueError(f'Unsupported missing value handling strategy: {strategy}')
    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df