from src.retrieve_data import read_params
import argparse
import pickle


def predict(data, config_path):
    config = read_params(config_path) 

    preprocessor_path = config['data_transformation']['preprocessor_path']
    model_dir_path = config['model']['saved_model']

    preprocessor = pickle.load(open(preprocessor_path, 'rb'))
    model = pickle.load(open(model_dir_path, 'rb'))
    
    # Use the preprocessor to transform the data
    preprocessed_data = preprocessor.transform(data)
    
    # Predict using the model
    prediction = model.predict(preprocessed_data)
    
    # Convert the prediction to a human-readable format, e.g., 'Income: >50K'
    prediction_result = 'Income greater than $50,000' if prediction[0] else 'Income less than or equal to $50,000'
    
    return prediction_result


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_path=parsed_args.config