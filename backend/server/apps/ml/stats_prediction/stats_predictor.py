import joblib, time, torch
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import destandardize
import standardize as s


class StatsPredictor:
    def __init__(self):
        """
        Load preprocessing objects and Transformer object
        
        """
        self.model = joblib.load("transformer_model.joblib")
        self.active_player_dict = players.get_active_players()
        self.inputs = []
        self.current_input = None
        self.seasons = ['2020-21', '2021-22']
        
    def preprocessing(self, input_name):
        """
        Apply preprocessing to input data

        """
        self.inputs.append(input_name)
        input_name = input_name.lower()
        try:
            player = [player for player in self.active_player_dict if 
                      (player['full_name']).lower() == input_name][0]
            self.current_input = player
        except IndexError:
            print("Player does not exist or is not currently active. Please try another name.")
            return
    
        def game_to_month(player_id, season):
            """
            If no monthly stats can be found, turn each game
            stats into monthly stats.
        
            """
            gamelog = playergamelog.PlayerGameLog(player_id, season=season, season_type_all_star='Regular Season')
            df = gamelog.player_game_log.get_data_frame()
            time.sleep(0.5)
            df_np = df.to_numpy()
            months = [d[:3] for d in df_np[:,3]]
            stats = df_np[:,[6,7,8,13,14,18,19,20,21,22,23,24]]
            df_np = np.c_[stats, months]
            new_df = pd.DataFrame(df_np, columns=('MP','FGM','FGA','FTM','FTA','REB','AST','STL','BLK','TOV','PF','PTS','Month'))
            df = new_df.groupby(['Month'],sort=False).mean(False)
            return df[::-1]
        
        input_stats =  torch.tensor(game_to_month(self.current_input['id'], self.seasons[0]).values, 
                                    game_to_month(self.current_input['id'], self.seasons[1]).values)
        
        return s.standardize(input_stats)
    
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
    