from flask import Flask, render_template, request
import yaml
import pickle
import pandas as pd
from src.logger import logging

app=Flask(__name__)

params_path = "params.yaml"

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config['model']['saved_model']
    model = pickle.load(open(model_dir_path, 'rb'))
    prediction = model.predict(data)
    
    # Convert the prediction to a human-readable format, e.g., 'Income: >50K'
    prediction_result = 'Income greater than $50,000' if prediction[0] else 'Income less than or equal to $50,000'
    
    return prediction_result


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    prediction_result = None  # Initialize prediction result
    config = read_params(params_path)
    
    if request.method == 'POST':
        try:
            if request.form:
                # Extract input data from the form
                data = pd.DataFrame({
                    'age': [int(request.form['age'])],
                    'workclass_group': [int(request.form['workclass_group'])],
                    'fnlwgt': [int(request.form['fnlwgt'])],
                    'education_group': [request.form['education_group']],
                    'native_group': [int(request.form['native_group'])],
                    'hours-per-week': [int(request.form['hours-per-week'])],
                    'race_group': [int(request.form['race_group'])],
                    'sex_group': [int(request.form['sex_group'])],
                    'marital_status_group': [int(request.form['marital_status_group'])],
                    'occupation_group': [request.form['occupation_group']],
                    'Relationship_Group': [request.form['Relationship_Group']],
                    'capital-gain': [int(request.form['capital-gain'])],
                    'capital-loss': [int(request.form['capital-loss'])]
                })
                
                # Load the preprocessor from the configuration
                preprocessor_path = config['data_transformation']['preprocessor_path']
                preprocessor = pickle.load(open(preprocessor_path, 'rb'))
                
                # Transform the input data using the preprocessor
                data_scaled = preprocessor.transform(data)
                logging.info('"Data received and processed successfully."')
                
                # Make the prediction
                prediction_result = predict(data_scaled)
    
        except Exception as e:
            error = {"error": "Oops, something went wrong. Please try again."}
            logging.error("An error occurred: %s", str(e))
            return render_template("error.html", error=error)
    
    return render_template('form.html', prediction_result=prediction_result)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=8080)