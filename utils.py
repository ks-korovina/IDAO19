import os
from itertools import repeat
import numpy as np
import pandas as pd

# a folder with initial csv data and where is hits features will be created
PATH = ""

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']

TRAIN_COLUMNS = ["label", "weight"]

FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_Z", "FOI_hits_T", 
              "FOI_hits_S",
                "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_DZ", "FOI_hits_DT"]

ID_COLUMN = "id"


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)


def load_full_train_csv(nrows=5445705, path=PATH):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    train = pd.read_csv(os.path.join(path, "train.csv"),
                       converters=converters,
                       dtype=types,
                       nrows=nrows)
    return train

def load_full_test_csv(path=PATH):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv(os.path.join(path, "test_public.csv.gz"),
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test

def load_full_test_final_csv():
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv("../data/IDAO/test_private_v2_track_1.csv",
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test




def find_closest_hit_per_station(row):
    result = np.empty(20, dtype=np.float32)
    result.fill(np.nan)

    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    mean_dist_per_station = result[8:12]
    mean_dist_sqrt_per_station  = result[12:16]
    median_dist_per_station  = result[16:20]

    
    for station in range(4):
        hits = (row["FOI_hits_S"] == (station + 1))
        if hits.any():
            x_distances_2 = (row["Lextra_X[%i]" % station] - row["FOI_hits_X"][hits])**2
            y_distances_2 = (row["Lextra_Y[%i]" % station] - row["FOI_hits_Y"][hits])**2
            distances_2 = x_distances_2 + y_distances_2
            closest_hit = np.argmin(distances_2)

            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            mean_dist_per_station[station] = np.mean(distances_2)
            mean_dist_sqrt_per_station[station] = np.mean(np.sqrt(distances_2))
            median_dist_per_station[station] = np.median(distances_2)

    return result