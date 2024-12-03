import pickle
import os
from data_processing import prepare_data, preprocess_text_nltk
from features import feature
from modeling import modeling
from inference import predict_polarity

# Set environment variables for file paths (to be used inside the Docker container)
amazon_data_path = os.getenv('AMAZON_DATA_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Sentiment_Analysis/data/test_amazon.csv')
amazon_test_path = os.getenv('AMAZON_TEST_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Sentiment_Analysis/data/test_amazon.csv')
movie_data_path = os.getenv('MOVIE_DATA_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Sentiment_Analysis/data/train.csv')
model_save_path = os.getenv('MODEL_SAVE_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Sentiment_Analysis/models/sample_trained_model.pickle')

# Clean data
data, train_text, test_text, train_labels, test_labels = prepare_data(amazon_data_path, amazon_test_path, movie_data_path)

print("Data Summary", data.groupby(["reviewtype", "polarity", "datatype"]).count())

# Preprocess the training and test data
train_text_processed = preprocess_text_nltk(train_text)
test_text_processed = preprocess_text_nltk(test_text)
print("Data cleaned..")

# Features
max_feature_num = 500
train_vec, test_vec, vectorizer = feature(train_text_processed, test_text_processed, max_feature_num)
print("Vectorized..")

# Modeling
clf = modeling(train_vec, train_labels, test_vec, test_labels)
print("Model fitted..")

# Save model and other necessary modules
all_info_want_to_save = {
    'model': clf,
    'vectorizer': vectorizer
}
with open(model_save_path, "wb") as save_path:
    pickle.dump(all_info_want_to_save, save_path)
print(f"Model saved to {model_save_path}..")

# inference
sentiment = predict_polarity(model_save_path,)

