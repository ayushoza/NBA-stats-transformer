#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 03:18:36 2022

@author: janelle
"""
import joblib
import pandas as pd

class StatsPredictor:
    def __init__(self):
        """
        Load preprocessing objects and Transformer object
        
        """
        path_to_model = ""
        self.model = joblib.load("transformer_model.joblib")
        
    def preprocessing(self, input_data):
        """
        Turn JSON to pandas DataFrame and apply preprocessing

        """
        pass
    
    def predict(self, input_data):
        """
        Predict stats using the transformer model object

        """
        pass
    
    def postprocessing(self):
        """
        Apply post-processing on prediction values

        """
        pass
    
    def compute_prediction(self):
        """
        Combine preprocessing, predict and postprocessing
        and return JSON object with response
        
        """
        pass
    