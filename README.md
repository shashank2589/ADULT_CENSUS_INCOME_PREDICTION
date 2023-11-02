# Income Prediction App

This project demonstrates the development of an income prediction application. The application predicts whether an individual's income is above or below $50,000 based on various features.

## Setup Virtual Environment
```
conda create -p env python=3.8
```
## Activate the enviroment
```
source activate ./env (git bash)
```
First, you need to install the libraries required for your project. You can create a requirements.txt file that lists all the dependencies

## Install Necessary Libraries in your virtual environment using
```
pip install -r requirements.txt
```
## Build the Package:
To build the Python package, use the following command from the project's root directory
```
python setup.py
```
## Version Control
```
git init
```
```
dvc init
```
```
dvc add data .
```
```
git add .
```
```
git commit -m "initial commit"
```
```
git push -u origin main

```
## Data Retrieval
To retrieve data from MongoDB, use the retrieve_data.py script. MongoDB connection details in params.yaml.
```
python retrieve_data.py
```
## Data Transformation
Data preprocessing and transformation are performed to prepare the data for model training. Data cleaning, feature engineering, and encoding are part of this step.
```
python data_transformation.py
```
## Model Training
A machine learning model is trained using the transformed data. Various models and hyperparameters are tested to optimize model performance.
```
python model_training.py
```
## DVC.yaml
The dvc.yaml file contains information about the data and stages in the DVC pipeline. You can reproduce the pipeline using DVC by running:
```
dvc repro
```
## Web Application
A web application is created using Flask to deploy the trained model. The app provides an interface for users to input data and get predictions.
```
python app.py
```
## Using MLflow to Choose the Best Model
MLflow is used to track and manage machine learning experiments. It helps select the best-performing model based on metrics like ROC AUC score.

## Creating the MLflow Artifacts Folder
MLflow artifacts, including models, metrics, and parameters, are logged to a folder named mlflow_artifacts for later retrieval.
```
mkdir mlflow_artifacts
```
## Running MLflow Server
You can run the MLflow server using the following command:
```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow_artifacts \
    --host localhost -p 1234
```
## Docker Image
A Docker image is created to package the Flask app and its dependencies. You can build the Docker image using:
```
docker build -t income-prediction:latest .
```
## CI/CD Workflows
CI/CD workflows are set up to deploy the model to AWS. The deployment process involves pushing the Docker image to Amazon Elastic Container Registry (ECR), and then deploying the application on Amazon Elastic Compute Cloud (EC2).

The CI/CD pipeline automates these steps and ensures smooth model deployment to AWS.

The CI/CD workflows for this project are defined in the .github/workflows directory. They automatically build and deploy the Docker image to AWS ECR when changes are pushed to the main branch.

