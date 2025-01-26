#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("ufcmaster.csv")
required_cols = [
    "RedOdds",
    "BlueOdds",
    "TitleBout",
    "WeightClass",
    "NumberOfRounds",
    "BlueCurrentLoseStreak",
    "BlueCurrentWinStreak",
    "BlueAvgSigStrPct",
    "BlueAvgSigStrLanded",
    "BlueAvgTDLanded",
    "BlueAvgTDPct",
    "BlueLosses",
    "BlueTotalRoundsFought",
    "BlueTotalTitleBouts",
    "BlueWinsByDecisionMajority",
    "BlueWinsByDecisionSplit",
    "BlueWinsByDecisionUnanimous",
    "BlueWinsByKO",
    "BlueWinsBySubmission",
    "BlueWins",
    "BlueHeightCms",
    "BlueReachCms",
    "RedOdds",
    "RedCurrentLoseStreak",
    "RedCurrentWinStreak",
    "RedAvgSigStrPct",
    "RedAvgSigStrLanded",
    "RedAvgTDLanded",
    "RedAvgTDPct",
    "RedLosses",
    "RedTotalRoundsFought",
    "RedTotalTitleBouts",
    "RedWinsByDecisionMajority",
    "RedWinsByDecisionSplit",
    "RedWinsByDecisionUnanimous",
    "RedWinsByKO",
    "RedWinsBySubmission",
    "RedWins",
    "RedHeightCms",
    "RedReachCms",
    "BetterRank",
    "TotalFightTimeSecs",
    "Winner",
    "Finish",
    "FinishRound",
]
df = df[required_cols]
df = df[df["Finish"] != "DQ"]

#maps

WeightClassMap = {
    "Women's Strawweight":1,
    "Women's Flyweight":2,
    "Women's Bantamweight":3,
    "Women's Featherweight":4,
    "Flyweight":5,
    "Bantamweight":6,
    "Featherweight":7,
    "Lightweight":8,
    "Welterweight":9,
    "Middleweight":10,
    "Light Heavyweight":11,
    "Heavyweight":12,
}

BetterRankMap = {
    "Blue": 0,
    "Red": 1,
    "neither": 2
}

FinishMap = {
    "U-DEC": 0,
    "S-DEC": 1,
    "KO-TKO": 2,
    "SUB": 3,
}

df["Winner"] = (df["Winner"] == "Red").astype(int) #red is 1 and blue is 0
df["TitleBout"] = (df["TitleBout"] == "TRUE").astype(int) #True is 1 and False is 0
df["WeightClass"] =  df["WeightClass"].map(WeightClassMap)
df["BetterRank"] =  df["BetterRank"].map(BetterRankMap)
df["Finish"] =  df["Finish"].map(FinishMap)

# 1. Replace empty strings "" with NaN
df.replace("", np.nan, inplace=True)

# 2. Now fill all NaN values with 0
df.fillna(0, inplace=True)

train, valid, test = np.split(df.sample(frac=1) , [int(0.6*len(df)), int(0.8*len(df))])
#[int(0.6*len(df)), int(0.8*len(df))] -> everything from 0 to 60% is going to be the train dataset, everything from 60% to 80% is going
#to be the validitation dataset and everything from 80% to 100% is going to be the test dataset.

#function to seperate the x and y
import copy
def get_xy(dataframe, y_label, x_labels=None): #set x_labels to none by default, so we can get all or only the specified ones
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None: #if x_labels was not specified
    x = dataframe[[c for c in dataframe.columns if c != y_label]].values #get all the features that are not the same as the y_label
  else: #if x_labels was specified
    if len(x_labels) == 1: #if only one was chosen
      x = dataframe[x_labels].values.reshape(-1, 1) #set x to the values at the specified feature and reshape to 2D
    else: #if more than one label was specified
      x = dataframe[x_labels].values  #set x to the values of the specified features
  y = dataframe[y_label].values.reshape(-1, 1) #set y to the values of the specified label and reshape it to 2D
  data = np.hstack((x,y)) #set data to a horixontally stacked x and y
  return data, x, y #return the dataset, the x values, and the y values

_, x_train_winner, y_train_winner = get_xy(train, "Winner", x_labels=required_cols[:-3])
_, x_test_winner, y_test_winner = get_xy(test, "Winner", x_labels=required_cols[:-3])
_, x_valid_winner, y_valid_winner = get_xy(valid, "Winner", x_labels=required_cols[:-3])

_, x_train_finish, y_train_finish = get_xy(train, "Finish", x_labels=required_cols[:-3])
_, x_test_finish, y_test_finish = get_xy(test, "Finish", x_labels=required_cols[:-3])
_, x_valid_finish, y_valid_finish = get_xy(valid, "Finish", x_labels=required_cols[:-3])

_, x_train_finishRound, y_train_finishRound = get_xy(train, "FinishRound", x_labels=required_cols[:-3])
_, x_test_finishRound, y_test_finishRound = get_xy(test, "FinishRound", x_labels=required_cols[:-3])
_, x_valid_finishRound, y_valid_finishRound = get_xy(valid, "FinishRound", x_labels=required_cols[:-3])

from joblib import load
import pandas as pd

finish = load("finish_model.joblib")
finsihRound = load("finishRound_model.joblib")
winner = load("winner_model.joblib")
from sklearn.metrics import classification_report
print(classification_report(y_test_finish, finish.predict(x_test_finish)))
print(classification_report(y_test_finishRound, finish.predict(x_test_finishRound)))
print(classification_report(y_test_winner, finish.predict(x_test_winner)))