import os
from io import StringIO
import math


import numpy as np
import pandas as pd
import scipy
import mpmath as mp
import multiprocess


COUNT_INT = np.uint64
if COUNT_INT == np.uint32:
    # HIGHEST_COUNT = np.uint32(2 ** 32 - 1)
    HIGHEST_COUNT = np.uint32(2 ** 24 - 1)
elif COUNT_INT == np.uint64:
    # HIGHEST_COUNT = np.uint64(2 ** 64 - 1)
    # HIGHEST_COUNT = np.uint64(2 ** 32 - 1)
    HIGHEST_COUNT = np.uint64(2 ** 24 - 1)


def csv_to_df(file_name, dtype=COUNT_INT):
    with open(file_name, 'r') as f:
        text = f.read()

    if text[0] == '\t':
        # Removing the initial separator, if any, as it makes problems for pd.read_csv
        text = text[1:]

    dtypes = {}
    for column in pd.read_csv(
            StringIO(text),
            sep='\t',
        nrows=0).columns:
        dtypes[column] = dtype

    return pd.read_csv(
        StringIO(text),
        sep='\t',
        dtype=dtypes,
    )


def get_size_factors(df):
    _data = df.values
    N = _data.shape[1]
    _rows = []
    for _row in _data:
        if np.all(_row != 0):
            _rows.append(_row)
    J = len(_rows)
    data = np.array(_rows)
    data.shape = (J, N)
    counts_normalized = np.zeros(data.shape, dtype=np.float64)
    for j in range(data.shape[0]):
        _row = np.array(data[j, :])
        # counts = np.array([max(1, count) for count in row])
        counts = np.array([count for count in _row if count != 0])
        # print(counts)

        # Geometric mean for one gene (all samples)
        if len(counts) > 0:
            # denominator = math.exp(np.mean(np.log(counts)))
            denominator = mp_gmean(counts)
            # print(counts)
            # denominator = reduce(operator.mul, counts, 1) ** (1 / len(counts))
            # denominator = reduce(lambda x, y: x*y, counts)**(1.0/len(counts))
            # print(denominator)
        else:
            denominator = 0

        if denominator == 0:
            counts_normalized[j] = np.zeros(data.shape[1], dtype=np.float64)
        else:
            counts_normalized[j] = mp_fdiv(_row, denominator)
    # print(counts_normalized)
    size_factors = np.zeros(data.shape[1], dtype=np.float64)
    for i in range(data.shape[1]):
        column = np.array([count for count in counts_normalized[:, i] if count != 0])
        size_factors[i] = np.median(column)
    return size_factors


def save_df_to_csv(data_df, file_name):
    if not os.path.exists(file_name):
        data_df.to_csv(file_name, sep='\t')

        with open(file_name, 'r') as f:
            # Removing the initial separator, as it makes problems for pd.read_csv
            text = f.read()[1:]

        with open(file_name, 'w') as f:
            f.write(text)
    else:
        print('The file', file_name, 'already exists, not saving...')


def save_dfz_to_csv(dfz, filename):
    save_df_to_csv(dfz, filename)
    dfp = pd.DataFrame(convert_zscores_to_pvalues(dfz.values), index=dfz.index, columns=dfz.columns)
    filename_pv = os.path.splitext(filename)[0] + '-pv' + os.path.splitext(filename)[1]
    save_df_to_csv(dfp, filename_pv)


def convert_zscores_to_pvalues(zs__):
    return 2 * scipy.stats.norm.cdf(-np.abs(zs__))


def clean_zs(data):
    _tmp = np.copy(data)
    _tmp[np.isinf(_tmp)] = 0
    _tmp[np.isneginf(_tmp)] = 0
    data[np.isinf(data)] = max(7, np.abs(_tmp).max())
    data[np.isneginf(data)] = min(-7, -np.abs(_tmp).max())
    return data


mp_power = np.frompyfunc(mp.power, 2, 1)
mp_fdiv = np.frompyfunc(mp.fdiv, 2, 1)


def mp_gmean(array):
    return mp_power(mp_fprod((mp.mpf(str(e)) for e in array)), (1.0 / len(array)))


def mp_fprod2(a, b):
    return mp.fprod([a, b])


def mp_fprod(list_):
    f = np.frompyfunc(mp_fprod2, 2, 1)
    res = mp.mpf('1')
    for e in list_:
        res = f(res, e)
    return res


def transform(a, transform_f, axis=None, print_=False, mp=True):
    # a = a.astype(np.float64)
    # a += 0.001
    a_new = np.zeros_like(a, dtype=np.float64)
    if len(a.shape) == 1 or axis is None:
        shape = a.shape
        a = np.ravel(a)
        a_new = transform_f(a)
        a_new.shape = shape
        return a_new
    elif axis == 0:  # Column-wise'
        def f(c, column):
            if print_:
                print('Processing column', c)
            return c, transform_f(column)
        if mp:
            # multiprocessing
            # n_parts = math.floor(multiprocess.cpu_count() * 3 / 4) or 1
            n_parts = math.floor(multiprocess.cpu_count() * 5 / 12) or 1
            with multiprocess.Pool(processes=n_parts) as pool:
                results = pool.starmap(f, zip(range(a.shape[1]), (a[:, c] for c in range(a.shape[1]))))
            results.sort()
            for c in range(a.shape[1]):
                a_new[:, c] = results[c][1]
        else:
            # Single-processing
            for c in range(a.shape[1]):
                if print_:
                    print('Processing column', c)
                a_new[:, c] = transform_f(a[:, c])
        return a_new
    elif axis == 1:  # Row-wise
        def f(r, row):
            if print_:
                print('Processing row', r)
            return r, transform_f(row)
        if mp:
            # multiprocessing
            # n_parts = math.floor(multiprocess.cpu_count() * 3 / 4) or 1
            n_parts = math.floor(multiprocess.cpu_count() * 5 / 12) or 1
            with multiprocess.Pool(processes=n_parts) as pool:
                results = pool.starmap(f, zip(range(a.shape[0]), (a[r, :] for r in range(a.shape[0]))))
            results.sort()
            for r in range(a.shape[0]):
                a_new[r, :] = results[r][1]
        else:
            # Single-processing
            for r in range(a.shape[0]):
                if print_:
                    print('Processing row', r)
                a_new[r, :] = transform_f(a[r, :])
        return a_new
    else:
        raise Exception('Cannot happen')


def _std(data):
    assert len(data.shape) == 1
    N = len(data)
    data = clean_zs(data)
    # mu = data.mean()
    mu = np.median(data)
    try:
        c4 = np.sqrt(2/(N - 1)) * math.gamma(N/2) / math.gamma((N - 1)/2)
    except OverflowError: # N too big
        c4 = 1 - 1/4/N - 7/32/(N**2) - 19/128/(N**3)
    # std = data.std(ddof=1) / c4
    std = 1.4826 * mad(data) / c4
    # std = np.sqrt(((data - mu) ** 2).sum() / (data.size - 1))
    if std == 0:
        std = 0.000000000000001

    return std


def std(data, axis=None):
    return transform(data, _std, axis=axis, mp=False)


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def _standardize(data):
    assert len(data.shape) == 1
    N = len(data)
    data = clean_zs(data)
    mu = data.mean()
    # mu = np.median(data)
    try:
        c4 = np.sqrt(2/(N - 1)) * math.gamma(N/2) / math.gamma((N - 1)/2)
    except OverflowError: # N too big
        c4 = 1 - 1/4/N - 7/32/(N**2) - 19/128/(N**3)
    std = data.std(ddof=1) / c4
    # std = 1.4826 * mad(data) / c4
    # std = np.sqrt(((data - mu) ** 2).sum() / (data.size - 1))
    if std == 0:
        std = 0.000000000000001

    return (data - mu) / std
