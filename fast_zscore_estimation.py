import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
import scipy

import helpers as h


# THRESHOLD = 3.17
THRESHOLD = 0


def get_z_scores(data, l_ji_other__=None):
    J = data.shape[0]
    N = data.shape[1]

    try:
        c4 = np.sqrt(2 / (N - 1)) * math.gamma(N / 2) / math.gamma((N - 1) / 2)
    except OverflowError:  # N too big
        c4 = 1 - 1 / 4 / N - 7 / 32 / (N ** 2) - 19 / 128 / (N ** 3)
    mu__ = np.nanmean(data, axis=1)
    mu__.shape = (data.shape[0], 1)

    l_ji__ = np.log2((data + 1) / (mu__ + 1))
    l_j_ = np.nanmean(l_ji__, axis=1)
    l_j_.shape = (data.shape[0], 1)
    l_j_std_ = np.nanstd(l_ji__, axis=1, ddof=1) / c4
    # l_j_std_[l_j_std_ == 0] = 0.001
    l_j_std_[l_j_std_ == 0] = 0.000000000000001
    l_j_std_.shape = (data.shape[0], 1)

    if l_ji_other__ is None:
        l_ji_other__ = l_ji__
    z_scores = (l_ji_other__ - l_j_) / l_j_std_

    return z_scores, l_ji__


def run(data_file):
    # dl = h.DataLoader(outpyr.settings, data_file)
    # df = dl.data_normalized_sf
    # data_cleaned = df.values.astype(np.float64)
    df =h.csv_to_df(data_file, dtype=np.float64)
    data = df.values.astype(np.float64)/h.get_size_factors(df).reshape((1, df.values.shape[1]))
    data_cleaned = np.array(data)

    z_scores, l_ji__ = get_z_scores(data)
    # print(np.any(np.abs(z_scores) > THRESHOLD))
    z_scores, _ = get_z_scores(data_cleaned, l_ji__)

    fname = os.path.splitext(data_file)[0] + '-fzse-zs.csv'
    h.save_dfz_to_csv(pd.DataFrame(z_scores, index=df.index, columns=df.columns), fname)
    return os.path.abspath(fname)


def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')
    args = parser.parse_args()
    # print(args.corrected)
    data_file = args.data_file[0]

    run(data_file)


if __name__ == '__main__':
    main()
