import os
import sys
import argparse
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from retrieve_data import read_params, retrieve_data
from sklearn.model_selection import train_test_split


def save_and_split_data(config_path):
    logging.info('Save and split data method Starts')
    try:
        config = read_params(config_path)
        df = retrieve_data(config_path)
        raw_data_path = config["save_and_split_data"]["raw_dataset"]
        df.to_csv(raw_data_path, index=False)
        train_data_path = config["save_and_split_data"]["train_path"]
        test_data_path = config["save_and_split_data"]["test_path"]
        test_size = config["save_and_split_data"]["test_size"]
        random_state = config["save_and_split_data"]["random_state"]

        df = pd.read_csv(raw_data_path, sep=",")
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=random_state)
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        logging.info('Ingestion of Data is completed')
    except Exception as e:
        logging.info('Exception occured at Data Ingestion stage')
        raise CustomException(e, sys)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    save_and_split_data(config_path=parsed_args.config)
