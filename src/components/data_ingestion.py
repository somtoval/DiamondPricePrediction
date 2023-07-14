import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

## intitialize the Data Ingestion configuration

# The dataclass decorator
# It simplifies the process of creating classes that are primarily used to store data by automatically generating common boilerplate code
@dataclass
# Creating a class called DataIngestionconfig
class DataIngestionconfig:    
    # The class has attribute train_data_path which is annotated as string and is equal to the part artifacts/train.csv
    train_data_path:str=os.path.join('artifacts','train.csv')

    # The class has attribute test_data_path which is annotated as string and is equal to the part artifacts/test.csv
    test_data_path:str=os.path.join('artifacts','test.csv')

    # The class has attribute raw_data_path which is annotated as string and is equal to the part artifacts/raw.csv
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## Creating the data ingestion class
class DataIngestion:

    # We defined the class init method without arguments, so it does not take inputs directly
    def __init__(self):
        # we created an instance variable "ingestion_config" and set it as an object of DataIngestionconfig() class
        self.ingestion_config=DataIngestionconfig()

    # Creating a method, it takes no argments
    def initiate_data_ingestion(self):
        # Logging our progress
        logging.info('Data Ingestion method starts')

        try:
            # We read our data which is stored in notebooks/data
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            # Logging our progress
            logging.info('Dataset read as pandas Dataframe')

            # Taking our the directory name using the instance variable that is the object of the DataIngestion calss and taking out the raw_data_path, then creating the path using makedirs() and if it exists we won't recreate it
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            # We are then saving our data to the "raw_data_path" atribute of the ingestion_config object
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            # Logging our progress
            logging.info('Raw data is created')

            # Splitting our dataset in train and testset
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            # Saving the train to the path specified by the attribute of the ingestion_config object
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # Saving the test to the path specified by the attribute of the ingestion_config object
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            # Logging our progress
            logging.info('Ingestion of Data is completed')

            # The method when called will return the train_data_path attribute which is where the train_set was saved and also return teh test_data_path which is where the test set was saved
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        # Our exception handling
        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)

    