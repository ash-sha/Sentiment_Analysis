import unittest
from sklearn.naive_bayes import MultinomialNB
from ..src.modeling import modeling

class TestModeling(unittest.TestCase):
    def test_train_model_with_mlflow(self):
        train_vec = [[1, 0], [0, 1]]
        train_labels = ["Positive", "Negative"]
        test_vec = [[1, 0]]
        test_labels = ["Positive"]

        clf = modeling(train_vec, train_labels, test_vec, test_labels)
        self.assertIsInstance(clf, MultinomialNB, "Model is not of expected type")

if __name__ == '__main__':
    unittest.main()
