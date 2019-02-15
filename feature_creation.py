"""
Feature generation module. All functions are now pd-vectorized.

"""

import numpy as np
import pandas as pd
import swifter


def add_nshared_eq_zero(df):
    df["NShared_eq_zero"] = (df.NShared == 0).astype(int)

#def parse_FOI(df):
#     FOI_S_concatenated = np.concatenate(df.FOI_hits_S.values)
#     FOI_hits_X_flat = np.concatenate(df.FOI_hits_X.values)
#     FOI_hits_Y_flat = np.concatenate(df.FOI_hits_Y.values)
#     split_by = np.cumsum(df.FOI_hits_N.values.astype(int))[:-1]
    
#     for i in range(1, 5):
#         FOI_hits_X_flat_cur = FOI_hits_X_flat.copy()
#         FOI_hits_X_flat_cur[FOI_S_concatenated != i] = np.nan
        
#         df["FOI_hits_X_%i" % i] = np.split(FOI_hits_X_flat_cur, split_by) - df["Lextra_X[%i]" % (i-1)]
        
#         FOI_hits_X_flat_cur = FOI_hits_Y_flat.copy()
#         FOI_hits_X_flat_cur[FOI_S_concatenated != i] = np.nan
#         df["FOI_hits_Y_%i" % i] = np.split(FOI_hits_X_flat_cur, split_by) - df["Lextra_Y[%i]" % (i-1)]

# def add_nanmean_and_median(df):
#     for coor in ["X", "Y"]:
#         for i in range(1, 5):
#             df["FOI_hits_%s_%i_mean" % (coor, i)] = df["FOI_hits_%s_%i" % (coor, i)].swifter.apply(lambda x: np.nanmean(x))
#             df["FOI_hits_%s_%i_median" % (coor, i)] = df["FOI_hits_%s_%i" % (coor, i)].swifter.apply(lambda x: np.nanmedian(x))


def add_distance_to_closest(df):
    for i in range(4):
        df["dx_%i_squared" % i] = (df["Lextra_X[%i]" % i] - df["MatchedHit_X[%i]" % i])**2
        df["dy_%i_squared" % i] = (df["Lextra_Y[%i]" % i] - df["MatchedHit_Y[%i]" % i])**2
        df["dxy_%i_squared" % i] = df["dy_%i_squared" % i] + df["dx_%i_squared" % i]


def add_momentum_features(df):
    df["P_forward"] = np.sqrt(df.P**2 - df.PT**2)
    df["tan_momentum"] = df.PT / df.P_forward


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def numpy_angle_dist(x, y):
    a = (x - y) / 3.1416
    
    cond = np.where(a > 1)
    a[cond] = (a - 2)[cond]
    
    cond = np.where(a < -1)
    a[cond] = (a + 2)[cond]
    return a


def add_coordinates_features(df):
    X_TO_Y_RATIO = 1.23

    df['MatchedHit_X[0]'] /= X_TO_Y_RATIO
    df['MatchedHit_X[1]'] /= X_TO_Y_RATIO
    df['MatchedHit_X[2]'] /= X_TO_Y_RATIO
    df['MatchedHit_X[3]'] /= X_TO_Y_RATIO

    df["dist_0"], df["phi_0"] = cart2pol(df["MatchedHit_X[0]"], df["MatchedHit_Y[0]"])
    df["dist_1"], df["phi_1"] = cart2pol(df["MatchedHit_X[1]"], df["MatchedHit_Y[1]"])
    df["dist_2"], df["phi_2"] = cart2pol(df["MatchedHit_X[2]"], df["MatchedHit_Y[2]"])
    df["dist_3"], df["phi_3"] = cart2pol(df["MatchedHit_X[3]"], df["MatchedHit_Y[3]"])
    
    df["dist_ratio_10"] = df.dist_1 / df.dist_0
    df["dist_ratio_21"] = df.dist_2 / df.dist_1 
    df["dist_ratio_32"] = df.dist_3 / df.dist_2

    df["dist_ratio_mean"] = np.mean(np.array([df["dist_ratio_10"].values,
                                              df["dist_ratio_21"].values,
                                              df["dist_ratio_32"].values]), axis=0)

    df["dist_ratio_std"] = np.std(np.array([df["dist_ratio_10"].values,
                                              df["dist_ratio_21"].values,
                                              df["dist_ratio_32"].values]), axis=0)

    
    df["phi_diff_10"] = numpy_angle_dist(df.phi_0.values, df.phi_1.values)
    df["phi_diff_21"] = numpy_angle_dist(df.phi_1.values, df.phi_2.values)
    df["phi_diff_32"] = numpy_angle_dist(df.phi_2.values, df.phi_3.values)
    
    
    df["phi_diff_mean"] = np.abs(np.mean(np.array([df["phi_diff_10"].values,
                                              df["phi_diff_21"].values,
                                              df["phi_diff_32"].values]), axis=0))

    df["phi_diff_std"] = np.std(np.array([df["phi_diff_10"].values,
                                              df["phi_diff_21"].values,
                                              df["phi_diff_32"].values]), axis=0)

    
    df["path_rho"], df["path_phi"] = cart2pol(df["Lextra_X[3]"] - df["Lextra_X[0]"], 
                                              df["Lextra_Y[3]"] - df["Lextra_Y[0]"])
    
    
    intermediate_features_to_drop = [
       'phi_0', 'dist_1', 'phi_1', 'dist_2', 'phi_2',
       'dist_3', 'phi_3',
        "dist_ratio_10", "dist_ratio_21", "dist_ratio_32",
        "phi_diff_10", "phi_diff_21", "phi_diff_32"
    ]
    df.drop(columns=intermediate_features_to_drop, inplace=True)
    


# def compute_d2_with_closest(row):
#     # TODO

#     df.loc[:, "d2_matched"] = 0.
#     for station in range(4):
#         lextra_x, lextra_y = df["Lextra_X[{}]".format(station)],\
#                              df["Lextra_Y[{}]".format(station)]
    
#         # TODO: actual closest from all FOI: CHANGE THE NAME HERE
#         closest_x, closest_y = df["closest_x_{}".format(station)], \
#                                df["closest_y_{}".format(station)]
        
#         x_dist = closest_x - lextra_x
#         y_dist = closest_y - lextra_y
#         pad_x = df["MatchedHit_DX[{}]".format(station)]
#         pad_y = df["MatchedHit_DY[{}]".format(station)]
#         df["d2_matched"] += (x_dist/pad_x) ** 2 + (y_dist/pad_y) ** 2
#     df["d2_matched"] /= 4
#     return df


def compute_d2_with_closest(df):
    # ready
    df.loc[:, "d2_closest"] = 0.
    for station in range(4):
        closest_x2, closest_y2 = df["closest_x_{}".format(station)], \
                                 df["closest_y_{}".format(station)]
        pad_x = df["MatchedHit_DX[{}]".format(station)]
        pad_y = df["MatchedHit_DY[{}]".format(station)]
        df["d2_closest"] += closest_x2 / pad_x ** 2 + closest_x2 / pad_y ** 2
    df["d2_closest"] /= 4
    return df

def compute_d2_with_mean(df):
    # ready
    df.loc[:, "d2_mean"] = 0.
    for station in range(4):
        closest_x2, closest_y2 = df["closest_x_{}".format(station)], \
                                 df["closest_y_{}".format(station)]
        pad_x = df["MatchedHit_DX[{}]".format(station)]
        pad_y = df["MatchedHit_DY[{}]".format(station)]
        df["d2_mean"] += closest_x2 / pad_x ** 2 + closest_x2 / pad_y ** 2
    df["d2_mean"] /= 4
    return df

def add_na_at3(df):
    # ready
    df.loc[:, "isna_at3"] = df["closest_x_2"].isna()
    return df

def add_na_at4(df):
    # ready
    df.loc[:, "isna_at4"] = df["closest_x_3"].isna()
    return df

def compute_d2_with_matchedhit(df):
    # ready
    
    df.loc[:, "d2_matched"] = 0.
    for station in range(4):
        lextra_x, lextra_y = df["Lextra_X[{}]".format(station)],\
                             df["Lextra_Y[{}]".format(station)]
        
        # MatchedHit as closest
        closest_x, closest_y = df["MatchedHit_X[{}]".format(station)], \
                               df["MatchedHit_Y[{}]".format(station)]
        
        x_dist = closest_x - lextra_x
        y_dist = closest_y - lextra_y
        pad_x = df["MatchedHit_DX[{}]".format(station)]
        pad_y = df["MatchedHit_DY[{}]".format(station)]
        df["d2_matched"] += (x_dist/pad_x) ** 2 + (y_dist/pad_y) ** 2
    df["d2_matched"] /= 4
    return df

def compute_binary_hits(df, max_dist2_list):
    """
    Use max_dist to cut off hits that are far from Lextra;
    
    # ~30% cutoffs for every station
    max_dist2_list = [3000, 5000, 10000, 11000]
    
    # 20% cutoffs for every station
    max_dist2_list = [5669, 9213, 18248, 29150]
    
    # 10% cutoffs for every station (corresponding to ~real proportion of non-muons)
    max_dist2_list = [12273, 20032, 39633, 74037]

    *** Apply: ***
    binary_hits = small_train.apply(lambda row: compute_binary_hits(row, max_dist2_list),
                                    result_type="expand", axis=1)
    
    """
    hits_s = df["FOI_hits_S"]
    #hits_s = list_from_str(hits_s, int)
    momentum = df["P"] / 10**5
    
    has_hit = np.zeros((df.shape[0], 4))
    for s in range(4):
        #dist_to_closest = (df["Lextra_X[%i]" % s] - df["MatchedHit_X[%i]" % s])**2
        #dist_to_closest += (df["Lextra_Y[%i]" % s] - df["MatchedHit_Y[%i]" % s])**2
        dist_to_closest = df["closest_x_{}".format(s)]
        dist_to_closest += df["closest_y_{}".format(s)]
        has_hit[:, s] = (dist_to_closest < max_dist2_list[s])

    one_and_two = np.logical_and(has_hit[:, 0], has_hit[:, 1])
    three_or_four = np.logical_or(has_hit[:,2], has_hit[:,3])
    three_and_four = np.logical_and(has_hit[:,2], has_hit[:,3])
    
    res = np.where(momentum <= 6., one_and_two,
                   np.logical_and(one_and_two, three_or_four))
    return np.where(momentum <= 10., res,
                    np.logical_and(one_and_two, three_and_four))

