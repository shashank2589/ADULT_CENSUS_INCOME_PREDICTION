from flask import Flask, render_template, request
import pandas as pd
from prediction_pipeline.prediction import predict
from src.logger import logging

app = Flask(__name__)

# Specify the path to the configuration file
config_path = "params.yaml"

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    # Assign a default value to prediction_result
    prediction_result = None

    if request.method == 'GET':
        return render_template("form.html")

    try:
        if request.method == 'POST' and request.form:
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

            logging.info('Data received and processed successfully.')

            # Make the prediction
            prediction_result = predict(data, config_path=config_path)

    except Exception as e:
        error = {"error": str(e)}
        logging.error("An error occurred: %s", str(e))
        return render_template("error.html", error=error)

    return render_template('result.html', prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8080)
