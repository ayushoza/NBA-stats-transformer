#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 03:18:36 2022

@author: janelle
"""
import joblib
import pandas as pd
from nba_api.stats.static import players

class StatsPredictor:
    def __init__(self):
        """
        Load preprocessing objects and Transformer object
        
        """
        path_to_model = ""
        self.model = joblib.load("transformer_model.joblib")
        self.active_player_dict = players.get_active_players()
        
    def preprocessing(self, input_name):
        """
        Apply preprocessing to input data

        """
        input_name = input_name.title()
        try:
            player = [player for player in self.active_player_dict if 
                      (player['full_name']).lower() == input_name][0]
            return player['id']
        except IndexError:
            print("Player does not exist or is not currently active. Please try another name.")
        return 
    
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
    