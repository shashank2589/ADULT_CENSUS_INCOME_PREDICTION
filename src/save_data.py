import os
import sys
import argparse
from src.exception import CustomException
from src.logger import logging
from retrieve_data import read_params, retrieve_data

def save_data(config_path):
    logging.info('Save data method Starts')
    try:
        config = read_params(config_path)
        df = retrieve_data(config_path)
        raw_data_path = config["save_data"]["raw_dataset"]
        df.to_csv(raw_data_path, index=False)
        logging.info('Ingestion of Data is completed')
    except Exception as e:
        logging.info('Exception occured at Data Ingestion stage')
        raise CustomException(e,sys)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    save_data(config_path=parsed_args.config)