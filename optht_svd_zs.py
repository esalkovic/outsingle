import argparse
import os
from pprint import pprint

import pandas as pd
import scipy
# import matplotlib.pyplot as plt
import numpy as np

import helpers as h
import optht

LINE = '----------------------------------'


# standardize = h.standardize
def standardize(data, axis=None):
    return h.transform(data, h._standardize, axis=axis, print_=False, mp=False)


def svd(data, U, s, VT, s_dimension):
    S = scipy.linalg.diagsvd(np.pad(s[:s_dimension], (0, len(s) - s_dimension)), *data.shape)
    return U.dot(S.dot(VT))


def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')
    args = parser.parse_args()
    # print(args.corrected)
    data_file = args.data_file[0]
    process(data_file)


def process(data_file):
    out_basename = os.path.splitext(data_file)[0] + '-svd-optht'
    out_name = out_basename + '-zs.csv'
    if os.path.isfile(out_name):
        print('File', out_name, 'already exists.')
        return out_name
    df = h.csv_to_df(data_file, dtype=np.float64)

    data_original = h.clean_zs(df.values)

    for transpose in (False, True):
        if transpose:
            data = data_original.transpose()
        else:
            data = data_original
        U, s, VT = scipy.linalg.svd(data)

        s_dimension = optht.optht(data, sv=s, sigma=None)
        if transpose:
            print('TRANSPOSE!')
        # print('OPTHT rank:', s_dimension)

        data_new = svd(data, U, s, VT, s_dimension)

        stds = h.std(data_new, axis=1)[:, 0].reshape((data_new.shape[0], 1))
        zs_outrider__ = (data - data_new) / stds
        _data2 = standardize(zs_outrider__, axis=1)

        # zs_outrider_ideal__ = data - data_new
        # print(h.std(zs_outrider_ideal__, axis=1)[:, 0])
        # _data2 = standardize(zs_outrider_ideal__, axis=1)

        # _data2 = standardize(data - data_new, axis=None)
        if transpose:
            # h.save_dfz_to_csv(pd.DataFrame(data_new.transpose(), index=df.index, columns=df.columns), out_basename + '-signal-t-zs.csv')
            # h.save_dfz_to_csv(pd.DataFrame((data - data_new).transpose(), index=df.index, columns=df.columns), out_basename + '-won-t-zs.csv')
            h.save_dfz_to_csv(pd.DataFrame(_data2.transpose(), index=df.index, columns=df.columns), out_basename + '-t-zs.csv')
        else:
            # h.save_dfz_to_csv(pd.DataFrame(data_new, index=df.index, columns=df.columns), out_basename + '-signal-zs.csv')
            # h.save_dfz_to_csv(pd.DataFrame(data - data_new, index=df.index, columns=df.columns), out_basename + '-won-zs.csv')
            h.save_dfz_to_csv(pd.DataFrame(_data2, index=df.index, columns=df.columns), out_name)
        # h.save_dfz_to_csv(pd.DataFrame(zs_outrider_ideal__, index=df.index, columns=df.columns), out_basename + '-ori-zs.csv')

    return out_name


if __name__ == '__main__':
    main()
