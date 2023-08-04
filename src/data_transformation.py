import sys
import argparse
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from retrieve_data import read_params
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
import pickle


def data_transformation(config_path):
    logging.info('Data Transformation initiated')
    try:
        config = read_params(config_path)
        train_data_path = config["save_and_split_data"]["train_path"]
        test_data_path = config["save_and_split_data"]["test_path"]
        target_col = config["project"]["target_col"]
        preprocessor_path = config["data_transformation"]["preprocessor_path"]
        # Reading train and test data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        logging.info('Read train and test data completed')
        logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
        logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

        # Split the data into Independent and dependent features
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        train_X = train_df.drop(target_col, axis=1)
        test_X = test_df.drop(target_col, axis=1)

        # Divide the columns into categorical and numerical
        numerical_cols = config["data_transformation"]["numerical_cols"]
        categorical_cols = config["data_transformation"]["categorical_cols"]
        categorical_cols1 = config["data_transformation"]["categorical_cols1"]
        # Define the custom ranking for ordinal variable
        edu_category = config["data_transformation"]["edu_category"]

        logging.info('Pipeline Initiated')

        # Numerical Pipeline
        num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                       ('scaler', StandardScaler())])

        # Categorigal Pipeline
        cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('ordinalencoder', OrdinalEncoder(categories=[edu_category])),])

        cat_pipeline1 = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehotencoder', OneHotEncoder(handle_unknown="ignore")),])

        # Combine
        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols),
            ('cat_pipeline1', cat_pipeline1, categorical_cols1)
        ])

        logging.info('Pipeline Completed')
        # Transforming using preprocesso
        logging.info("Applying preprocessor on training and testing datasets.")
        X_train = pd.DataFrame(preprocessor.fit_transform(train_X))
        X_test = pd.DataFrame(preprocessor.transform(test_X))

        with open(preprocessor_path, "wb") as file_obj:
            pickle.dump(preprocessor, file_obj)
        logging.info('Preprocessor pickle file saved')

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.info("Error in Initiating Data Transformation Stage")
        raise CustomException(e, sys)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data_transformation(config_path=parsed_args.config)
