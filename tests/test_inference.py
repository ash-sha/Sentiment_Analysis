import unittest
from ..src.inference import predict_polarity

class TestInference(unittest.TestCase):
    def test_predict_polarity(self):
        model_path = "models/sample_trained_model.pickle"
        user_query = "This is fantastic!"

        prediction = predict_polarity(model_path, user_query)
        self.assertIn(prediction, ["Positive", "Negative"], "Prediction should be valid polarity")

if __name__ == '__main__':
    unittest.main()
