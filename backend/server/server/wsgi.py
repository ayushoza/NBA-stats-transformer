"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.stats_prediction.stats_predictor import StatsPredictor

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    sp = StatsPredictor()
    # add to ML registry
    registry.add_algorithm(endpoint_name="stats_predictor",
                            algorithm_object=sp,
                            algorithm_name="transformer",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Janelle+Ayush",
                            algorithm_description="Transformer with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(StatsPredictor))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
