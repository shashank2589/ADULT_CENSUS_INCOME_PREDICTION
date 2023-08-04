import sys
import argparse
from retrieve_data import read_params
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_transformation import data_transformation
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import xgboost as xgb
import warnings
import json
warnings.filterwarnings('ignore')


def scores_evaluation(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    auc = roc_auc_score(actual, predicted)
    return accuracy, precision, recall, f1, auc


def model_trainer(config_path):
    logging.info('Model Trainer initiated')
    try:
        config = read_params(config_path)
        model_path = config["model"]["saved_model"]
        X_train, X_test, y_train, y_test = data_transformation(config_path)
        models = {
            'Decision_Tree': DecisionTreeClassifier(min_samples_leaf=3, max_depth=7, min_samples_split=2),
            'Random_Forest': RandomForestClassifier(n_estimators=300, min_samples_split=10, max_depth=10),
            'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=3, max_depth=7, min_samples_split=2), n_estimators=100),
            'XGBoost': xgb.XGBClassifier(n_estimators=149, max_depth=5, learning_rate=0.053823386514435315),
            'Logistic_Regression': LogisticRegression()
        }

        train_report = {}
        test_report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]
            # Train model
            model.fit(X_train, y_train)
            # Predict Training data
            y_train_pred = model.predict(X_train)
            # Predict Testing data
            y_test_pred = model.predict(X_test)
            # Get Accuracy scores for train data
            train_model_score = accuracy_score(y_train, y_train_pred) * 100
            # Get Accuracy scores for test data
            test_model_score = accuracy_score(y_test, y_test_pred) * 100

            train_report[model_name] = train_model_score.round(2)
            test_report[model_name] = test_model_score.round(2)
        print(train_report)
        print(test_report)
        print('\n', '='*50, '\n')
        logging.info(f'Train Model Report : {train_report}')
        logging.info(f'Test Model Report : {test_report}')

        # To get best model score from dictionary
        best_model_score = max(test_report.values())

        best_model_name = list(test_report.keys())[
            list(test_report.values()).index(best_model_score)]

        best_model = models[best_model_name]
        predicted = best_model.predict(X_test)
        print(
            f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score.round(2)} percent')
        print('\n', '='*50, '\n')
        logging.info(
            f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score.round(2)} percent')

        (accuracy, precision, recall, f1, auc) = scores_evaluation(
            y_test, predicted=predicted)

        metrics_file = config["reports"]["scores"]
        with open(metrics_file, "w") as f:
            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1 Score": f1,
                "roc_auc_score": auc
            }
            json.dump(scores, f)
        logging.info('Metrics Scores file saved')

        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)
        logging.info('Model pickle file saved')

        return train_report, test_report, best_model_name, best_model_score
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_trainer(config_path=parsed_args.config)
