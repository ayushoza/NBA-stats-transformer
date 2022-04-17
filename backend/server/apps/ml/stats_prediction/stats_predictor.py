import joblib
import time
import torch
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import torch.nn as nn
import os

here = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(here, "transformer_model.joblib")

std = torch.Tensor([9.49077602, 2.49329645, 5.03377988, 1.76282567, 2.21402973, 2.97681828,
                    2.14723644, 0.6380834,  0.64613585, 0.99951063, 0.92877199, 6.6417578])
mean = torch.Tensor([25.60003559,  4.29149587,  9.21989404,  2.16928589,  2.85579326,  4.67167195,
                     2.49971274,  0.86798461,  0.53945552,  1.58682799,  2.31876199, 11.31548302])


def destandardize(output):
    """
    Destandardize prediction to training data mean and standard deviation.
    
    * If any points are <0, then set to 0.

    """
    relu = nn.ReLU()
    d = relu((output * std) + mean)
    rounded = (d * 10).round() / 10
    return rounded

def standardize(input_stats):
    """
    Standardize prediction to training data mean and standard deviation.
    
    * If any points are <0, then set to 0.

    """
    relu = nn.ReLU()
    s = relu((input_stats - mean) / std)
    rounded = (s * 10).round() / 10
    return rounded



class StatsPredictor:
    def __init__(self):
        """
        Load preprocessing objects and Transformer object

        """
        self.model = joblib.load(filename)
        self.active_player_dict = players.get_active_players()
        self.inputs = []
        self.current_input = None
        self.seasons = ['2020-21', '2021-22']
        self.prediction = {}

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
            print(
                "Player does not exist or is not currently active. Please try another name.")
            return

        def game_to_month(player_id, season):
            """
            If no monthly stats can be found, turn each game
            stats into monthly stats.

            """
            gamelog = playergamelog.PlayerGameLog(
                player_id, season=season, season_type_all_star='Regular Season')
            df = gamelog.player_game_log.get_data_frame()
            time.sleep(0.5)
            df_np = df.to_numpy()
            months = [d[:3] for d in df_np[:, 3]]
            stats = df_np[:, [6, 7, 8, 13, 14, 18, 19, 20, 21, 22, 23, 24]]
            df_np = np.c_[stats, months]
            new_df = pd.DataFrame(df_np, columns=(
                'MP', 'FGM', 'FGA', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Month'))
            df = new_df.groupby(['Month'], sort=False).mean(False)
            return df[:6][::-1]

        # If player is a rookie, just duplicate this seasons' stats into a second season for simplicity
        if game_to_month(self.current_input['id'], self.seasons[0]).empty:
            season1 = torch.tensor(game_to_month(self.current_input['id'], self.seasons[1]).values)
            input_stats = torch.cat((season1, season1))
        else:
            season0 = torch.tensor(game_to_month(self.current_input['id'], self.seasons[0]).values)
            season1 = torch.tensor(game_to_month(self.current_input['id'], self.seasons[1]).values)
            input_stats = torch.cat((season0, season1))

        return standardize(input_stats)

    def predict(self, input_data):
        """
        Predict stats using the transformer model object

        """
        return destandardize(self.model(torch.unsqueeze(input_data.float(),0), torch.zeros(6, 12, 12))[-1])
        # FIX THIS

    def postprocessing(self, output, player_name):
        """
        Apply post-processing on prediction values

        """
        if player_name not in self.prediction:
            self.prediction[player_name] = {'MP': output[:, 0].detach().tolist(), 'FGM': output[:, 1].detach().tolist(), 'FGA': output[:, 2].detach().tolist(), 'FTM': output[:, 3].detach().tolist(),
                                            'FTA': output[:, 4].detach().tolist(), 'REB': output[:, 5].detach().tolist(), 'AST': output[:, 6].detach().tolist(), 'STL': output[:, 7].detach().tolist(),
                                            'BLK': output[:, 8].detach().tolist(), 'TOV': output[:, 9].detach().tolist(), 'PF': output[:, 10].detach().tolist(), 'PTS': output[:, 11].detach().tolist()}
        return self.prediction[player_name]

    def compute_prediction(self, input_data):
        """
        Combine preprocessing, predict and postprocessing

        """
        try:
            preprocess = self.preprocessing(input_data)
            pred = self.predict(preprocess)
            postprocess = self.postprocessing(pred, input_data)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return postprocess
    

