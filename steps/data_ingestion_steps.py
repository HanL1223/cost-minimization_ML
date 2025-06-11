import pandas as pd
import os
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path:str) ->pd.DataFrame:
    """
    Ingest data using DataIngestorFactory
    """
    file_extension = os.path.splitext(file_path)[1]
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    return df
