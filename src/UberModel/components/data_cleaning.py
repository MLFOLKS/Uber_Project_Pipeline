from abc import abstractmethod, ABC
from src.UberModel.config.logger import logging
from src.UberModel.config.exception import CustomException
from src.UberModel.utils.common import save_object
import pandas as pd
from typing import Union, Tuple
import sys
from math import radians, sin, cos, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from dataclasses import dataclass
import os
from zenml import step


class DataHandler(ABC):
    """
        Abstract class for data handler
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        pass
    

@dataclass
class DataConfig:
    transformed_obj_file_path = os.path.join('artifacts', 'transformed.pkl')


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataHandler) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)


def distance_transform(longitude1, latitude1, longitude2, latitude2):
    """
    Calculate the distance between two sets of coordinates using the Haversine formula.

    Parameters:
        longitude1 (float): The longitude of the first set of coordinates.
        latitude1 (float): The latitude of the first set of coordinates.
        longitude2 (float): The longitude of the second set of coordinates.
        latitude2 (float): The latitude of the second set of coordinates.

    Returns:
        list: A list of distances between the pairs of coordinates.
    """
    travel_dist = []
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        travel_dist.append(c)
    return travel_dist
    
    
class DataPreprocessing(DataHandler):
    """
        Handle the data by performing various data cleaning and inserting missing values.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be handled.
        
        Returns:
            pd.DataFrame: The cleaned and processed DataFrame.
    """       
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Drop rows with missing values
            df = data.dropna()

            # Drop unnecessary columns
            df.drop(columns=['Unnamed: 0', 'key'], axis=1, inplace=True)
            
            # Remove duplicate rows
            df.drop_duplicates(inplace=True)
            
            # Filter out rows with latitude and longitude values outside the valid range
            df = df[
                (df['pickup_latitude'].between(-180, 180)) &
                (df['dropoff_latitude'].between(-180, 180)) &
                (df['pickup_longitude'].between(-90, 90)) &
                (df['dropoff_longitude'].between(-90, 90))
            ]
            
            # Convert pickup_datetime column to datetime type
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            
            # Extract year, month, weekday, and hour from pickup_datetime
            df['year'] = df['pickup_datetime'].dt.year
            df['month'] = df['pickup_datetime'].dt.month
            df['weekday'] = df['pickup_datetime'].dt.weekday
            df['hour'] = df['pickup_datetime'].dt.hour
            
            # Map month to quarterly segments
            df['Monthly_Quarter'] = df['month'].apply(
                lambda x: 'Q1' if x in [1, 2, 3] else 'Q2' if x in [4, 5, 6] else 'Q3' if x in [7, 8, 9] else 'Q4'
            )
            
            # Map hour to hourly segments
            df['Hourly_Segments'] = df['hour'].apply(
                lambda x: 'H1' if x in range(4) else 'H2' if x in range(4, 8) else 'H3' if x in range(8, 12) else 'H4' if x in range(12, 16) else 'H5' if x in range(16, 20) else 'H6'
            )
            
            # Drop unnecessary columns
            df.drop(['pickup_datetime', 'month', 'hour'], axis=1, inplace=True)
            
            # Calculate distance traveled using custom distance_transform function
            df['distance_traveled'] = distance_transform(
                df['pickup_longitude'].to_numpy(),
                df['pickup_latitude'].to_numpy(),
                df['dropoff_longitude'].to_numpy(),
                df['dropoff_latitude'].to_numpy()
            )
            
            # Round distance_traveled column to 3 decimal places
            df['distance_traveled'] = df['distance_traveled'].round(3)
            
            # Filter out rows with invalid fare_amount, passenger_count, and distance_traveled values
            df = df[(df['fare_amount'] > 0) & (df['passenger_count'] < df['passenger_count'].max()) & (df['distance_traveled'] > 0.1)]
            
            # Drop rows with zero latitude or longitude values
            df.drop(df[(df['pickup_longitude'] == 0) | (df['pickup_latitude'] == 0) | (df['dropoff_longitude'] == 0) | (df['dropoff_latitude'] == 0)].index, inplace=True)
            
            # Drop latitude and longitude columns
            df.drop(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], axis=1, inplace=True)
            
            # Encode Monthly_Quarter and Hour
            df.Monthly_Quarter = LabelEncoder().fit_transform(df.Monthly_Quarter)
            df.Hourly_Segments = LabelEncoder().fit_transform(df.Hourly_Segments)
            return df
        except Exception as e:
            logging.error("Error in data preprocessing")
            raise CustomException(e, sys) from e
        
class DataDivideStrategy(DataHandler):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data by splitting it into training and testing sets.

        Parameters:
            data (pd.DataFrame): The input data to be split.

        Returns:
            Union[pd.DataFrame, pd.Series]: The training and testing sets for X and y.
        """
        try:
            features =  [
                'passenger_count', 'year',
                'weekday', 'Monthly_Quarter', 
                'Hourly_Segments', 'distance_traveled'
                ]
            target = 'fare_amount'
            X, y = data[features], data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in data Divide Strategy")
            raise CustomException(e, sys) from e
        
        

class DataTransformer:
    """
    Handle the transformation of the input data.

    Parameters:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The testing data.

    Returns:
        Union[np.ndarray, np.ndarray]: The transformed training data and the transformed testing data.
    """
    def __init__(self,X_train: pd.DataFrame, X_test: pd.DataFrame):
        self.X_train = X_train
        self.X_test = X_test
    
    def handle_transform(self) -> Union[np.ndarray, np.ndarray]:
        try:
            save_path = DataConfig().transformed_obj_file_path
            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X_train)
            X_test = scaler.transform(self.X_test)
            save_object(save_path, scaler)
            return X_train, X_test
        except Exception as e:
            logging.error("Error in data transformer")
            raise CustomException(e, sys) from e
        
        
        
@step
def clean_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    try:
        preprocess_data_strategy = DataPreprocessing()
        data_cleaning = DataCleaning(data, preprocess_data_strategy)
        preprocessed_data = data_cleaning.handle_data()
        divided_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divided_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        data_transform = DataTransformer(x_train, x_test)
        X_train, X_test = data_transform.handle_transform()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in data cleaning steps")
        raise CustomException(e, sys) from e