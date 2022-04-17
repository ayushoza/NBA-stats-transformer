from django.test import TestCase

from apps.ml.stats_prediction.stats_predictor import StatsPredictor

class MLTests(TestCase):
    def test_transf_algorithm(self):
        input_data = "Lebron James"
        my_alg = StatsPredictor()
        response = my_alg.compute_prediction(input_data)
        self.assertTrue('MP' in response)
        self.assertFalse('status' in response)
        self.assertNotEqual(response['MP'], [])