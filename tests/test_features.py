import unittest
from ..src.features import feature

class TestFeatures(unittest.TestCase):
    def test_vectorize_text(self):
        train_text = ["sample text", "another sample"]
        test_text = ["test text"]

        train_vec, test_vec, vectorizer = feature(train_text, test_text)
        self.assertEqual(train_vec.shape[0], 2, "Training vector size mismatch")
        self.assertEqual(test_vec.shape[0], 1, "Test vector size mismatch")

if __name__ == '__main__':
    unittest.main()
