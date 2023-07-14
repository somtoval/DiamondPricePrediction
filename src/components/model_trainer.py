# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
# Creating a class named ModelTrainerConfig
class ModelTrainerConfig:
    # A class attribute which is set to 'artifacts/model.pkl'
    trained_model_file_path = os.path.join('artifacts','model.pkl')

# Creating the model Trainer class
class ModelTrainer:
    # The init method here takes no arguments
    def __init__(self):
        # An instance variable which is set to be an object of the ModelTrainerConfig class
        self.model_trainer_config = ModelTrainerConfig()

    # Defining a method that takes in train array and test array
    def initate_model_training(self,train_array,test_array):
        try:
            # Logging our progress
            logging.info('Splitting Dependent and Independent variables from train and test data')
            # Setting the X_train, y_train, X_test and y_test to there various data
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # A dictionary of the models to be used with the name as keys and the model as values
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor()
        }
            # Return model report using our evaluate_model class which we created in our utils.py
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # # To get best model score from dictionary 
            # best_model_score = max(sorted(model_report.values())) # I don't think there is a need to sort before taking out max value
            best_model_score = max(model_report.values())

            # Returning the best model name which did by returning the model name using the returned index of the model bestscore # READ FUTHER TO GRASP
            best_model_name = list(model_report.keys())[
                # from the list of the dictionary values we finde the index bearing the score = best_model_score
                list(model_report.values()).index(best_model_score)
            ]
            
            # Returning the best model from the models dictionary we created, note it will return the model value which is for example Lasso()
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            # Saving our mode using our save_object class in utils
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          
        # Our Exception
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)