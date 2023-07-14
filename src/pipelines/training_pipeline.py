import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# By doing this the modules we imported will run only when we call it
if __name__=='__main__':
    # Defines an object of the imported DataIngestion class
    obj=DataIngestion()
    # The DataIngestion class has a method called initiate_data_ingestion() which returns train data path and test data path
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    
    ## Triggering the data transformation component
    # Creating an object of DataTransformation class we imported
    data_transformation =  DataTransformation()
    # Passing in our train_data_path and test_data_path as input to the object we created which will return a train array and test array
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    
    ## Triggering the Model Trainer Component
    # Creating an object of the Model Trainer class 
    model_trainer=ModelTrainer()
    # passing in our train array and test array to the object's initiate_model_training() method
    model_trainer.initate_model_training(train_arr,test_arr)


