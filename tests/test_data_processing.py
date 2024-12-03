import unittest
from ..src.data_processing import prepare_data,preprocess_text_nltk

class TestDataProcessing(unittest.TestCase):
    def test_load_data(self):
        amazon_path = "/MLOps/Sentiment/data/test_amazon.csv"
        amazon_test_path = "/MLOps/Sentiment/data/test_amazon.csv"
        movie_path = "/MLOps/Sentiment/data/train.csv"
        data = prepare_data(amazon_path, amazon_test_path, movie_path)
        self.assertFalse(data.empty, "Loaded data should not be empty")

    def test_preprocess_text_nltk(self):
        sample_text = ["This is a Test!", "Another Sentence."]
        processed_text = preprocess_text_nltk(sample_text)
        self.assertEqual(processed_text, ["test", "another sentence"], "Preprocessing failed")

if __name__ == '__main__':
    unittest.main()
