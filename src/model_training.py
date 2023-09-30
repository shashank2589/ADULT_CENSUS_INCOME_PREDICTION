import sys
import argparse
import pickle
import json
import mlflow
from mlflow import sklearn
from retrieve_data import read_params
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_transformation import data_transformation
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import xgboost as xgb
import warnings
from src.exception import CustomException
from src.logger import logging

warnings.filterwarnings('ignore')

# Function to calculate evaluation metrics
def scores_evaluation(actual, predicted):
    accuracy = accuracy_score(actual, predicted) * 100
    precision = precision_score(actual, predicted) * 100
    recall = recall_score(actual, predicted) * 100
    f1 = f1_score(actual, predicted) * 100
    roc_auc = roc_auc_score(actual, predicted) * 100
    return accuracy, precision, recall, f1, roc_auc

# Main function for model training
def model_trainer(config_path):
    logging.info('Model Trainer initiated')
    
    try:
        config = read_params(config_path)
        X_train, X_test, y_train, y_test = data_transformation(config_path)
        mlflow_config = config["mlflow_config"]
        remote_server_uri = mlflow_config["remote_server_uri"]
        
        # Initialize MLflow with remote server URI
        mlflow.set_tracking_uri(remote_server_uri)
        
        # Set the experiment name
        mlflow.set_experiment(mlflow_config["experiment_name"])
        
        models = {
            'Decision_Tree': DecisionTreeClassifier(min_samples_leaf=1, max_depth=7, min_samples_split=2),
            'Random_Forest': RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=10),
            'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=1, max_depth=7, min_samples_split=2), n_estimators=100),
            'XGBoost': xgb.XGBClassifier(n_estimators=122, max_depth=3, learning_rate=0.4313788899225024, scale_pos_weight=2.6),
            'Logistic_Regression': LogisticRegression(),
            'Adaboost': AdaBoostClassifier(learning_rate=1.0, n_estimators=1000)
        }

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Log model name as a parameter
                mlflow.log_param("model_name", model_name)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict Testing data
                y_test_pred = model.predict(X_test)
                
                # Calculate and log evaluation metrics
                (accuracy, precision, recall, f1, roc_auc) = scores_evaluation(y_test, predicted=y_test_pred)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc_score", roc_auc)

                # Log the model to MLflow
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_trainer(config_path=parsed_args.config)
