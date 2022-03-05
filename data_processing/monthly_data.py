import pandas as pd
import numpy as np
import time
import ast
import csv
from nba_api.stats.endpoints import playergamelog

# format player_id and seasons to a dictionary from seasons_after.csv
player_df = pd.read_csv("seasons_after.csv", header=None)
player_df = player_df.rename(columns={0: 'Player ID', 1: 'Seasons'})
player = {}
for index, row in player_df.iterrows():
    s = list(dict.fromkeys(ast.literal_eval(player_df['Seasons'][index])))
    if not s == ['']:
        seasons = s
        player_ids = player_df['Player ID'][index]
        player[player_ids] = seasons


def player_month_id(player_id, month, season):
    """
    Generate an id with player id, season and month.
    format: PLAYER_ID-SEASON-MONTH

    >>> player_month_id('2544', 'May', '2016-17')
    '2544-May-2016-17'
    """
    months_nums = {"JAN": "1", "FEB": "2", "MAR": "3", "APR": "4", "MAY": "5", "JUN": "6",
                   "JUL": "7", "AUG": "8", "SEP": "9", "OCT": "10", "NOV": "11", "DEC": "12"}
    return int(months_nums[month]+season[:4]+season[5:]+str(player_id))


def game_to_month(player_id, season):
    """
    If no monthly stats can be found, turn each game
    stats into monthly stats.
    """
    gamelog = playergamelog.PlayerGameLog(
        player_id, season=season, season_type_all_star='Regular Season')
    df = gamelog.player_game_log.get_data_frame()
    time.sleep(0.8)
    df_np = df.to_numpy()
    months = [d[:3] for d in df_np[:, 3]]
    stats = df_np[:, [6, 7, 8, 13, 14, 18, 19, 20, 21, 22, 23, 24]]
    pm_id = np.array([])

    # include the player_month_id to keep track of monthly information and averages
    for ms in months:
        pm_id = np.append(pm_id, player_month_id(player_id, ms, season))
    df_np = np.c_[stats, pm_id, months]
    np.set_printoptions(threshold=5000)

    # list of all required statistics
    new_df = pd.DataFrame(df_np, columns=('MP', 'FGM', 'FGA', 'FTM', 'FTA',
                          'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'ID', 'Month'))
    for c in new_df:
        if c != "Month":
            new_df[c] = pd.to_numeric(new_df[c])

    # calculate the average for each monthly statistic
    df = new_df.groupby('Month', sort=False).agg('mean').round(3)
    return df[::-1]


# save progress for each run on the API to avoid losing progress after Timeout Error
player_stats = {}
file = open('seasons_after.csv', 'r')
total = len(file.readlines())
file.close()
curr = pd.read_csv("labelled_monthly.csv")
l = curr.shape[0]
l += 1
file = open('labelled_monthly.csv', 'a')
writer = csv.writer(file)

# write to labelled_monthly.csv
player_stats = {}
for p in range(len(list(player.keys())[l:])):
    plr = list(player.keys())[l:][p]
    frames = []
    for s in player[plr]:
        df_months = game_to_month(plr, s)
        frames.append(df_months)
    result = pd.concat(frames)
    player_stats[plr] = result.to_numpy()
    writer.writerow([plr, player_stats[plr]])
    print(total - l - p)
