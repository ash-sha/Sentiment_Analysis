import os
import pickle
from data_processing import preprocess_text_nltk

MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', '/MLOps/1.Sentiment_Analysis/models/sample_trained_model.pickle')

# Function to load the model and vectorizer, and make predictions
def predict_polarity(user_query: str) -> str:
    saved_model_dic = pickle.load(open(MODEL_SAVE_PATH, "rb"))
    saved_clf = saved_model_dic['model']
    saved_vectorizer = saved_model_dic['vectorizer']

    # Preprocess the user query
    preprocessed_query = preprocess_text_nltk([user_query])

    # Transform the query text using the saved vectorizer
    query_vec = saved_vectorizer.transform(preprocessed_query)

    # Predict the polarity
    prediction = saved_clf.predict(query_vec)

    return prediction[0]