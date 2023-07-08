import os
import sys
import yaml
import pandas as pd
import argparse
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from src.exception import CustomException
from src.logger import logging

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def retrieve_data(config_path):
    logging.info('Data Retrieving from MongoDB Database Starts')
    try:
        load_dotenv(find_dotenv())
        password = os.environ.get("mongodb_pwd")
        config = read_params(config_path)
        data_path = config["data_source"]["mongodb_url"]
        data_path = data_path.replace("{password}", password)
        db_name = config["data_source"]["database"]
        coll_name = config["data_source"]["collection"]

        
        with MongoClient(data_path) as client:
            db = client[db_name]
            coll = db[coll_name]
            dataset = coll.find()
            list_cursor = list(dataset)
            df = pd.DataFrame(list_cursor)
            df = df.drop("_id", axis=1)
        logging.info('Retrieval of Data is completed')
        return df

    except Exception as e:
        logging.info('Exception occured at Data Retrieval stage')
        raise CustomException(e,sys)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    retrieve_data(config_path=parsed_args.config)