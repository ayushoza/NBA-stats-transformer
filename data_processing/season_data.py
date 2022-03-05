from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import time
import csv
import re

csv_file = open('player_season.csv', 'r')

# retrieve all seasons that each player has played in from PlayerCareerStats endpoint
l = len(csv_file.readlines())
csv_file = open('player_season.csv', 'a')
writer = csv.writer(csv_file)
player_dict = players.get_players()
for i in player_dict[l:]:
    player_stats = playercareerstats.PlayerCareerStats(
        str(i['id']))
    player_seasons = player_stats.get_data_frames()[0]
    writer.writerow([i['id'], list(player_seasons['SEASON_ID'])])
    time.sleep(1)


# using player_season.csv to remove all player seasons before 1973-74 season
seasons_after = {}
for line in csv_file:
    line = line.strip().replace("[", "")
    line = line.replace("]", "")
    line = line.split(",")
    plyr = line[0]
    seasons_after[plyr] = []
    line = line[1:]
    for i in line:
        if not (re.search("19[0-6]", i) or re.search("197[0-3]", i)):
            i = i.strip('"')
            i = i.replace("'", "")
            i = i.replace('\\', '')
            i = i.replace(' ', '')
            seasons_after[plyr].append(i)
    if seasons_after[plyr] == []:
        seasons_after.pop(plyr)

# write to seasons_after.csv
with open('seasons_after.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in seasons_after.items():
        writer.writerow([key, value])
