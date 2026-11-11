import unittest

from diabetes_classifier import predict_risk


class DiabetesClassifierTests(unittest.TestCase):
    def test_prediction_has_valid_labels(self):
        predicted, actual = predict_risk(0)
        valid = {"higher-risk", "lower-risk"}
        self.assertIn(predicted, valid)
        self.assertIn(actual, valid)

    def test_invalid_index_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_risk(-1)


if __name__ == "__main__":
    unittest.main()
