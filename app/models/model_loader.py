import os
import joblib
from gensim.models.doc2vec import Doc2Vec


def load_model_and_vectorizer():
    # Define the file paths for the trained model and vectorizer
    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'logistic_regression_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'weights', 'tfidf_vectorizer.pkl')

    # Load the trained model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def load_doc2vec_model():
    # Define the file path for the trained Doc2Vec model
    doc2vec_model_path = os.path.join(os.path.dirname(__file__), 'weights', 'doc2vec_model_ag_news')

    # Load the trained Doc2Vec model
    doc2vec_model = Doc2Vec.load(doc2vec_model_path)

    return doc2vec_model