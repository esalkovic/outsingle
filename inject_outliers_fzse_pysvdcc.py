import math
import os
import datetime
import random

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
# import KDEpy

import helpers as h
import fast_zscore_estimation
import optht_svd_zs
import optht

now = datetime.datetime.now

np.random.seed(51007469)

nb = np.random.negative_binomial

OUTLIER_TYPES = (
    'b',
    'o',
    'u',
)

Z_SCORES = (
    # 1,
    # 2,
    # 3,
    # 4,
    #
    # 5,

    6,
    7,
    8,
    # 9,
    # 10,
)

# How many outliers per sample?
OUTLIER_FREQUENCY = 1

NOPYTHON = False

NUMBER_OF_TRIALS = 10


# standardize = h.standardize
def standardize(data, axis=None):
    return h.transform(data, h._standardize, axis=axis, print_=False, mp=False)


def svd(data, U, s, VT, s_dimension):
    S = scipy.linalg.diagsvd(np.pad(s[:s_dimension], (0, len(s) - s_dimension)), *data.shape)
    return U.dot(S.dot(VT))


def inject(fname, outlier_frequency, z_score, outlier_type):
    name, ext = os.path.splitext(fname)

    id_ = ('-wo-f%d-%s-z%.2f' % (outlier_frequency, outlier_type, z_score))

    df = h.csv_to_df(fname, dtype=np.float64)
    sf_ = h.get_size_factors(df).reshape((1, df.values.shape[1]))
    data__ = df.values.astype(np.float64)

    data_sf__ = data__/sf_
    J = data_sf__.shape[0]
    N = data_sf__.shape[1]

    try:
        c4 = np.sqrt(2 / (N - 1)) * math.gamma(N / 2) / math.gamma((N - 1) / 2)
    except OverflowError:  # N too big
        c4 = 1 - 1 / 4 / N - 7 / 32 / (N ** 2) - 19 / 128 / (N ** 3)

    mu__ = np.nanmean(data_sf__, axis=1)
    mu__.shape = (data_sf__.shape[0], 1)

    l_ji__ = np.log2((data_sf__ + 1) / (mu__ + 1))
    l_j_ = np.nanmean(l_ji__, axis=1)
    l_j_.shape = (data_sf__.shape[0], 1)
    l_j_std_ = np.nanstd(l_ji__, axis=1, ddof=1) / c4
    l_j_std_[l_j_std_ == 0] = 0.000000000000001
    l_j_std_.shape = (data_sf__.shape[0], 1)

    zs__ = (l_ji__ - l_j_) / l_j_std_

    # OHT
    U, s, VT = scipy.linalg.svd(zs__)

    s_dimension = optht.optht(zs__, sv=s, sigma=None)
    # Low-rank zs__ representation, i.e. de-noised
    zs_lr_optht__ = svd(zs__, U, s, VT, s_dimension)
    noise = zs__ - zs_lr_optht__

    zs_optht__ = np.empty_like(noise)
    mu_optht_ = np.empty(J)
    std_optht_ = np.empty(J)
    for j in range(J):
        data_j_ = noise[j, :]
        data_j_ = h.clean_zs(data_j_)
        mu = data_j_.mean()
        std = data_j_.std(ddof=1) / c4
        # std = np.sqrt(((data - mu) ** 2).sum() / (data.size - 1))
        if std == 0:
            std = 0.000000000000001

        zs_optht__[j, :] = (data_j_ - mu) / std
        mu_optht_[j] = mu
        std_optht_[j] = std

    print('mu_optht_', mu_optht_.mean())
    print('std_optht_', std_optht_.mean())
    # Create an outlier mask
    data_with_outliers = data__.astype(np.float64)
    outlier_mask = np.zeros_like(data__, dtype=np.int32)

    for i in range(N):
        ois = random.sample(range(J), outlier_frequency)
        for oi in ois:
            outlier_value = z_score
            if outlier_type == 'u':
                outlier_value = -outlier_value
            elif outlier_type == 'b':
                outlier_value = random.choice([-1, 1]) * outlier_value
            outlier_mask[oi, i] = outlier_value


    # Reuse zs_optht__ from the original data
    # zs_optht_wo__ = np.array(zs_optht__)

    # Generate zs_optht_wo__ from scratch as an ideal SND
    zs_optht_wo__ = np.random.normal(size=zs_optht__.shape)

    # print('zs_optht_wo__', zs_optht_wo__.std(), zs_optht_wo__.mean())
    # plt.hist(np.ravel(zs_optht_wo__), bins=1000);plt.show()
    zs_optht_wo__[outlier_mask != 0] = outlier_mask[outlier_mask != 0]

    noise_wo__ = zs_optht_wo__ * std_optht_.reshape((J, 1)) + mu_optht_.reshape((J, 1))
    zs_wo__ = zs_lr_optht__ + noise_wo__

    l_ji_wo__ = zs_wo__ * l_j_std_ + l_j_
    data_sf_wo__ = 2**l_ji_wo__ * (mu__ + 1) - 1

    data_wo__ = data_sf_wo__ * sf_
    data_wo__[data_wo__ < 0] = 0

    outlier_mask_df = pd.DataFrame(
        data=outlier_mask,
        index=df.index,
        columns=df.columns,
    )
    fname_new = name + id_ + ext
    fname_outliers_new = fname_new[:-len(ext)] + '-omask' + ext
    h.save_df_to_csv(outlier_mask_df, fname_outliers_new)
    #h.copy_mtime(fname_new, fname_outliers_new)

    # data_with_outliers[outlier_mask != 0] = data_wo__[outlier_mask != 0]
    data_with_outliers = data_wo__
    data_with_outliers[np.isnan(data_with_outliers)] = data__[np.isnan(data_with_outliers)]
    data_with_outliers_df = pd.DataFrame(
        data=np.rint(data_with_outliers).astype(h.COUNT_INT),
        index=df.index,
        columns=df.columns,
    )

    h.save_df_to_csv(data_with_outliers_df, fname_new)

    # data_wo_df = pd.DataFrame(
    #     data=np.rint(data_wo__).astype(h.COUNT_INT),
    #     index=df.index,
    #     columns=df.columns,
    # )
    #
    # h.save_df_to_csv(data_wo_df, name + id_ + '-wo' + ext)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Inject outliers into a dataset.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')
    args = parser.parse_args()

    data_file = args.data_file[0]
    # print(data_file)
    t1 = now()
    if os.path.isfile(data_file):
        for outlier_type in OUTLIER_TYPES:
            for z_score in Z_SCORES:
                print('==============================================')
                print('Injecting outliers with frequency %d, z-score %f (over-expressed/under-expressed/both?: %s)' % (
                OUTLIER_FREQUENCY, z_score, outlier_type))
                print('==============================================')
                inject(data_file, OUTLIER_FREQUENCY, z_score, outlier_type)
    else:
        parser.error("The file (%s) you provided does not exist." % data_file)
    t2 = now()
    # print(t2 - t1)


if __name__ == '__main__':
    main()
