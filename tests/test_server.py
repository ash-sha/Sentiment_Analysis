import unittest
from fastapi.testclient import TestClient
from ..src.server import app


class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_endpoint(self):
        response = self.client.post("/predict/", json={"text": "Amazing product!"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["polarity"], ["Positive", "Negative"])

if __name__ == '__main__':
    unittest.main()
