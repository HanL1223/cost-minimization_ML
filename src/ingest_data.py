import os
import zipfile
from abc import ABC,abstractmethod
import pandas as pd


#Define an abstract class for Data Ingestor
#New data ingestion should follow simliar pattern
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,csv_type:str) ->pd.DataFrame:
        """Abstract method to ingest data from a given file"""
        pass

#Ingestor for .csv file
class CSVIngestor(DataIngestor):
    def ingest(self,csv_type:str) ->pd.DataFrame:
        """Extract .csv file and return content as pandas DataFrame"""
        extracted_file = os.listdir(f"data/{csv_type}ing_raw")
        csv_file = [f for f in extracted_file if f.endswith('.csv')]

        if len(csv_file) == 0:
                raise FileNotFoundError(f"No CSV file found in the {csv_type}ing folder.")
        if len(csv_file) > 1:
                raise ValueError(f"Multiple CSV files found in the {csv_type}ing folder.")
        
        csv_file_path = os.path.join("data",f"{csv_type}ing_raw",csv_file[0])
        
        df =pd.read_csv(csv_file_path)
        return df
    

class ZIPIngestor(DataIngestor):
     def ingest(self,csv_type:str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""

        file_path = f"data/{csv_type}ing_raw"
        zip_file =[f for f in os.listdir(file_path) if f.endswith('.zip')]
        # Extract the zip file
        with zipfile.ZipFile(file_path+f"/{zip_file[0]}", "r") as zip_ref:
            zip_ref.extractall()
        
        extracted_file = os.listdir()
        csv_file = [f for f in extracted_file if f.endswith('.csv')]

        if len(csv_file) == 0:
                raise FileNotFoundError(f"No CSV file found in the {csv_type}ing folder.")
        if len(csv_file) > 1:
                raise ValueError(f"Multiple CSV files found in the {csv_type}ing folder.")
        
        csv_file_path = os.path.join("data",f"{csv_type}ing_raw",csv_file[0])

        df =pd.read_csv(csv_file_path)
        return df
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZIPIngestor()
        elif file_extension == ".csv":
             return CSVIngestor
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


if __name__ == "__main__":
    # # Specify the file path
    # file_path = "/Users/ayushsingh/Desktop/end-to-end-production-grade-projects/prices-predictor-system/data/archive.zip"

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]

    # # Get the appropriate DataIngestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # # Ingest the data and load it into a DataFrame
    # df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    # print(df.head())  # Display the first few rows of the DataFrame
    pass
