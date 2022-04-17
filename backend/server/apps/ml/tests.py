from django.test import TestCase
import inspect
from apps.ml.registry import MLRegistry

from apps.ml.stats_prediction.stats_predictor import StatsPredictor

class MLTests(TestCase):
    
    def test_transf_algorithm(self):
        input_data = "Lebron James"
        my_alg = StatsPredictor()
        response = my_alg.compute_prediction(input_data)
        self.assertTrue('MP' in response)
        self.assertFalse('status' in response)
        self.assertNotEqual(response['MP'], [])
        
    
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "stats_predictor"
        algorithm_object = StatsPredictor()
        algorithm_name = "transformer"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Janelle+Ayush"
        algorithm_description = "Transformer with simple pre- and post-processing"
        algorithm_code = inspect.getsource(StatsPredictor)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
