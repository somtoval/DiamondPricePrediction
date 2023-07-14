import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
# Creating a class called DataTransformationConfig
class DataTransformationConfig:
    # An attribute or variable which is set to artifacts/preprocessor.pkl
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

# Creating the DataTransformation class
class DataTransformation:
    # The init method takes no input
    def __init__(self):
        # An instance variable that is set to be an object of DataTransformationConfig class
        self.data_transformation_config=DataTransformationConfig()

    # Creating a method with no arguments
    def get_data_transformation_object(self):
        try:
            # Logging our progress
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            # Logging our progress
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            # Joining the pipelines using sklearn ColumnTransformer()
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            # Returning the preprocessor object, which is an instance of the ColumnTransformer class from scikit-learn. 
            return preprocessor

            # Logging our progress
            logging.info('Pipeline Completed')

        # Our Exception
        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    # Creating a method with 2 arguments(train_path, test_path)
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Logging our Progress
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            # assigning the get_data_transformation_object() which returns a preprocessor object to a variable
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            # Droping the price feature for the train input feature
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            # assigning the price feature to a variable
            target_feature_train_df=train_df[target_column_name]

            # Dropping the price feature for the test input
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            # Assigning the price feature to a variable
            target_feature_test_df=test_df[target_column_name]
            
            # Fiting and Transforming the train feature using the preprocessor_obj and this returns a numpy array
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # Transforming the test feature using the preprocesor_obj and this returns a numpy array
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Logging our progress
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            # Using the numpy .c_ it concatenates numpy array column wise, so we concatenate the preprocessed input faeture to the target_feature_train_df which is the price column in our dataset, we had to convert it first to np array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # Using the numpy .c_ it concatenates numpy array column wise, so we concatenate the preprocessed input faeture to the target_feature_test_df which is the price column in our dataset, we had to convert it first to np array
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Using our save_object class we created in utils we pass in our file path(it is artifacts/preprocessor.pkl as stated in the class attribute) and the obj which is the preprocessor obj and it saves it
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            # Logging our progress
            logging.info('Preprocessor pickle file saved')

            # The method returns the train arr which is the transformed input data concatenated with the target, test_arr which is the transformed input data concatenated with the target and the file path 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        # Our exceptionn
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)