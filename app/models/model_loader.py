import os
import joblib


def load_model_and_vectorizer():
    # Define the file paths for the trained model and vectorizer
    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'logistic_regression_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'weights', 'tfidf_vectorizer.pkl')

    # Load the trained model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
