import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import pymongo

class Mongo:
    def __init__(self, db_name, collection_name):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        try:
            self.client = pymongo.MongoClient("mongodb URL")
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logging.info("MongoDB Connection eastablised")
        except Exception as e:
            logging.error("An error occurred while connecting to MongoDB")
            raise CustomException(e,sys)