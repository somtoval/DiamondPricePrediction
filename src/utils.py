import os
import sys
import pickle
import numpy as np 
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

# Defining a function that takes in 2 arguments (file_path of object to be store and object to be store)
def save_object(file_path, obj):
    try:
        # extracts the file directory name from the file path
        dir_path = os.path.dirname(file_path)

        # makes a directory using the extracted directory name
        os.makedirs(dir_path, exist_ok=True)

        # opens the created file path and dumps the obj using pickle.dump()
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raising an exception based on our custom exception class we have created and imported, it takes in the exception and sys( to get more info about  it allows the exception to include information about the system state or the Python interpreter)
        raise CustomException(e, sys)
    
# Defining a function that takes in X_train,y_train,X_test,y_test,models(which I think is a dictionary with the name of the model being the key and the model object as the value)
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        # Creating an empty dictionary
        report = {}

        # A for loop that will run based on the length of the models dictionary
        for i in range(len(models)):
            # Returning the model object by turning the models.values() dictionary view object and taking the index of i
            model = list(models.values())[i]
            # Train model
            # Fitting the extracted model above to the x_train, and y_train data
            model.fit(X_train,y_train)

            

            # Predicting test data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            # We are creating a dictionary key name in our empty dictionary (report), the name is gotten from the models.keys() dictionary view object, we converted it to a list and took the ith index
            report[list(models.keys())[i]] =  test_model_score

        # We are turning the report dictionary, which will consist of differen models and their respective r2 scores
        return report

    # Exception Block
    except Exception as e:
        # Logging our exception
        logging.info('Exception occured during model training')
        # Raising an exception based on our custom exception class we have created and imported, it takes in the exception and sys( to get more info about  it allows the exception to include information about the system state or the Python interpreter)
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e, sys)