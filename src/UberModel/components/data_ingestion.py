from src.UberModel.config.exception import CustomException
from src.UberModel.config.logger import logging
import pandas as pd
import sys
from zenml import step
from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    """
        Data ingestion Config class which return file paths.
    """
    data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
        Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self):
        """Initialize the data ingestion class."""
        pass
    
    def get_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv("./data/uber.csv")
            logging.info("Data ingestion completed successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
        
        
@step
def ingest_data() -> pd.DataFrame:
    try:
        save_data_path = DataIngestionConfig().data_path
        data = DataIngestion().get_data()
        os.makedirs(os.path.dirname(
            save_data_path), exist_ok=True)
        data.to_csv(save_data_path,
                  index=False)
        return data
    except Exception as e:
            raise CustomException(e, sys) from e