# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    time_span = [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


if __name__ == "__main__":
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])