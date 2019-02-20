#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `dc_stat_think` package."""

import numpy as np
import pandas as pd
import scipy.stats as st

import pytest

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp

import dc_stat_think as dcst
import dc_stat_think.dc_stat_think as dcst_private
import dc_stat_think.original as original
import dc_stat_think.no_numba as no_numba

# 1D arrays for testing functions outside of edge cases
array_shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=10)
arrays = hnp.arrays(np.float, array_shapes, elements=hs.floats(-100, 100))

# 2D arrays for testing functions with two equal length input arrays
arrays_2 = hnp.arrays(np.float, (2, 10), elements=hs.floats(-100, 100))

# Tolerance on closeness of arrays
atol = 1e-10


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, arrays)
def test_ecdf_formal(x, data):
    correct = np.searchsorted(np.sort(data), x, side='right') / len(data)
    assert np.allclose(dcst.ecdf_formal(x, data), correct, atol=atol,
                       equal_nan=True)


def test_ecdf_formal_custom():
    assert dcst.ecdf_formal(0.1, [0, 1, 2, 3]) == 0.25
    assert dcst.ecdf_formal(-0.1, [0, 1, 2, 3]) == 0.0
    assert dcst.ecdf_formal(0.1, [3, 2, 0, 1]) == 0.25
    assert dcst.ecdf_formal(-0.1, [3, 2, 0, 1]) == 0.0
    assert dcst.ecdf_formal(2, [3, 2, 0, 1]) == 0.75
    assert dcst.ecdf_formal(1, [3, 2, 0, 1]) == 0.5
    assert dcst.ecdf_formal(3, [3, 2, 0, 1]) == 1.0
    assert dcst.ecdf_formal(0, [3, 2, 0, 1]) == 0.25

    with pytest.raises(RuntimeError) as excinfo:
        dcst.ecdf_formal([np.nan, np.inf], [0, 1, 2, 3])
    excinfo.match('Input cannot have NaNs.')

    correct = np.array([1.0, 1.0])
    result = dcst.ecdf_formal([3.1, np.inf], [3, 2, 0, 1])
    assert np.allclose(correct, result, atol=atol)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays)
def test_ecdf(data):
    x, y = dcst.ecdf(data)
    x_correct, y_correct = original.ecdf(data)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)
    assert np.allclose(y, y_correct, atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays_2, hs.integers(0, 1000000))
def test_swap_random(data, seed):
    a, b = data
    np.random.seed(seed)
    a_orig, b_orig = original.swap_random(a, b)
    dcst_private._seed_numba(seed)
    a_out, b_out = dcst.swap_random(a, b)

    assert len(a_out) == len(b_out) == len(a) == len(b)

    # Each entry should be present same number of times
    ab = np.sort(np.concatenate((a, b)))
    ab_out = np.sort(np.concatenate((a_out, b_out)))
    assert np.allclose(ab, ab_out, atol=atol, equal_nan=True)

    # Check for swaps matching
    for i in range(len(a)):
        ab = np.array([a[i], b[i]])
        ab_out = np.array([a_out[i], b_out[i]])
        assert ab[0] in ab_out
        assert ab[1] in ab_out


def test_ecdf_formal_for_plotting():
    data = np.array([2, 1, 3])
    y_correct = np.array([0, 0, 1, 1, 2, 2, 3, 3]) / 3
    x_correct = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    x, y = dcst.ecdf(data, formal=True, min_x=0, max_x=4)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)
    assert np.allclose(y, y_correct, atol=atol, equal_nan=True)

    data = np.array([1, 2, 2, 3])
    y_correct = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]) / 4
    x_correct = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 4])
    x, y = dcst.ecdf(data, formal=True, min_x=0, max_x=4)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)
    assert np.allclose(y, y_correct, atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, hs.integers(0, 1000000))
def test_bootstrap_replicate_1d(data, seed):
    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.mean)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.mean)
    assert (np.isnan(x) and np.isnan(x_correct, atol=atol, equal_nan=True)) \
                or np.isclose(x, x_correct, atol=atol, equal_nan=True)

    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.median)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.median)
    assert (np.isnan(x) and np.isnan(x_correct, atol=atol, equal_nan=True)) \
                or np.isclose(x, x_correct, atol=atol, equal_nan=True)

    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.std)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.std)
    assert (np.isnan(x) and np.isnan(x_correct, atol=atol, equal_nan=True)) \
                or np.isclose(x, x_correct, atol=atol, equal_nan=True)


def test_bootstrap_replicate_1d_nan():
    with pytest.raises(RuntimeError) as excinfo:
        dcst.bootstrap_replicate_1d(np.array([np.nan, np.nan]), np.mean)
    excinfo.match('Array must have at least 1 non-NaN entries.')


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, hs.integers(0, 1000000), hs.integers(1, 100))
def test_draw_bs_reps(data, seed, size):
    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.mean, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.mean, size=size)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)

    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.median, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.median,
                                      size=size)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)

    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.std, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.std, size=size)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)

    def my_fun(data):
        return np.dot(data, data)
    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, my_fun, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], my_fun, size=size)
    assert np.allclose(x, x_correct, atol=atol, equal_nan=True)


def test_draw_bs_pairs_linreg():
    for n in range(10, 20):
        for size in [1, 10, 100]:
            x = np.random.random(n)
            y = 1.5 * x + 3.0 + (np.random.random(n) - 0.5) * 0.1

            seed = np.random.randint(0, 100000)

            np.random.seed(seed)
            slope, intercept = no_numba.draw_bs_pairs_linreg(x, y, size=size)
            np.random.seed(seed)
            slope_correct, intercept_correct = \
                                original.draw_bs_pairs_linreg(x, y, size=size)
            assert np.allclose(slope, slope_correct, atol=atol,
                               equal_nan=True)
            assert np.allclose(intercept, intercept_correct, atol=atol, equal_nan=True)


def test_draw_bs_pairs_linreg_edge():
    x = np.ones(10)
    y = np.ones(10)
    slope, intercept = dcst.draw_bs_pairs_linreg(x, y, size=10)
    assert np.isnan(slope).all()
    assert np.isnan(intercept).all()



def test_draw_bs_pairs_linreg_nan():
    x = np.array([])
    y = np.array([])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.draw_bs_pairs_linreg(x, y, size=1)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([np.nan])
    y = np.array([np.nan])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.draw_bs_pairs_linreg(x, y, size=1)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([np.nan, 1])
    y = np.array([1, np.nan])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.draw_bs_pairs_linreg(x, y, size=1)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([0, 1, 5])
    y = np.array([1, np.inf, 3])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.draw_bs_pairs_linreg(x, y, size=1)
    excinfo.match('All entries in arrays must be finite.')


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays_2, hs.integers(0, 1000000), hs.integers(1, 100))
def test_draw_bs_pairs(data, seed, size):
    x, y = data
    def my_fun(x, y, mult):
        return mult * np.dot(x, y)
    def my_fun_orig(x, y):
        return 2.4 * np.dot(x, y)

    np.random.seed(seed)
    bs_reps = no_numba.draw_bs_pairs(x, y, my_fun, args=(2.4,), size=size)
    np.random.seed(seed)
    bs_reps_correct = original.draw_bs_pairs(x, y, my_fun_orig, size=size)
    assert np.allclose(bs_reps, bs_reps_correct, atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, arrays, hs.integers(0, 1000000))
def test_permutation_sample(data_1, data_2, seed):
    np.random.seed(seed)
    x, y = no_numba.permutation_sample(data_1, data_2)
    np.random.seed(seed)
    x_correct, y_correct = original.permutation_sample(data_1, data_2)
    assert np.allclose(x_correct, x, atol=atol, equal_nan=True)
    assert np.allclose(y_correct, y, atol=atol, equal_nan=True)

    x, y = dcst.permutation_sample(data_1, data_2)
    assert np.allclose(np.sort(np.concatenate((data_1, data_2))),
                       np.sort(np.concatenate((x, y))), atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, arrays, hs.integers(0, 1000000))
def test_draw_perm_reps(data_1, data_2, seed):
    # Have to use size=1 because np.random.shuffle and np.random.permutation
    # give different results on and after 2nd call
    np.random.seed(seed)
    x = no_numba.draw_perm_reps(data_1, data_2, no_numba.diff_of_means,
                                size=1)
    np.random.seed(seed)
    x_correct = original.draw_perm_reps(data_1, data_2, original.diff_of_means,
                                        size=1)
    assert np.allclose(x_correct, x, atol=atol, equal_nan=True)

    np.random.seed(seed)
    x = no_numba.draw_perm_reps(data_1, data_2,
                                no_numba.studentized_diff_of_means, size=1)
    np.random.seed(seed)
    x_correct = original.draw_perm_reps(
            data_1, data_2, no_numba.studentized_diff_of_means, size=1)
    assert np.allclose(x_correct, x, atol=atol, equal_nan=True)


    def my_fun(x, y, mult):
        return (np.mean(x) + np.mean(y)) * mult
    def my_fun_orig(x, y):
        return (np.mean(x) + np.mean(y)) * 2.4
    np.random.seed(seed)
    x = no_numba.draw_perm_reps(data_1, data_2, my_fun, args=(2.4,), size=1)
    np.random.seed(seed)
    x_correct = original.draw_perm_reps(data_1, data_2, my_fun_orig, size=1)
    assert np.allclose(x_correct, x, atol=atol, equal_nan=True)

    def diff_of_medians(data_1, data_2):
        return np.median(data_1) - np.median(data_2)
    np.random.seed(seed)
    x = no_numba.draw_perm_reps(data_1, data_2, diff_of_medians, size=1)
    np.random.seed(seed)
    x_correct = original.draw_perm_reps(data_1, data_2, diff_of_medians,
                                        size=1)
    assert np.allclose(x_correct, x, atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, arrays)
def test_diff_of_means(data_1, data_2):
    assert np.allclose(dcst.diff_of_means(data_1, data_2),
                      np.mean(data_1) - np.mean(data_2), atol=atol, equal_nan=True)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays, arrays)
def test_studentized_diff_of_means(data_1, data_2):
    if (np.allclose(data_1, data_1[0], rtol=1e-7, atol=1e-14)
          and np.allclose(data_2, data_2[0], rtol=1e-7, atol=1e-14)):
        assert np.isnan(dcst.studentized_diff_of_means(data_1, data_2))
    else:
        t, _ = st.ttest_ind(data_1, data_2, equal_var=False)
        assert np.isclose(dcst.studentized_diff_of_means(data_1, data_2), t)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays_2)
def test_pearson_r(data):
    x, y = data
    if np.allclose(x, x[0], atol=atol, equal_nan=True) or np.allclose(y, y[0], atol=atol, equal_nan=True):
        assert np.isnan(dcst.pearson_r(x, y))
    else:
        assert np.isclose(dcst.pearson_r(x, y), original.pearson_r(x, y))
        assert np.isclose(dcst.pearson_r(x, y), np.corrcoef(x, y)[0,1])


def test_pearson_r_edge():
    x = np.array([])
    y = np.array([])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.pearson_r(x, y)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([np.nan])
    y = np.array([np.nan])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.pearson_r(x, y)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([np.nan, 1])
    y = np.array([1, np.nan])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.pearson_r(x, y)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([0, 1, 5])
    y = np.array([1, np.inf, 3])
    with pytest.raises(RuntimeError) as excinfo:
        dcst.pearson_r(x, y)
    excinfo.match('All entries in arrays must be finite.')


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays)
def test_ks_stat(x):
    theor_data = np.random.normal(0, 1, size=100)
    correct, _ = st.ks_2samp(x, theor_data)
    assert np.isclose(dcst.ks_stat(x, theor_data), correct)

    theor_data = np.random.exponential(1, size=100)
    correct, _ = st.ks_2samp(x, theor_data)
    assert np.isclose(dcst.ks_stat(x, theor_data), correct)

    theor_data = np.random.logistic(0, 1, size=100)
    correct, _ = st.ks_2samp(x, theor_data)
    assert np.isclose(dcst.ks_stat(x, theor_data), correct)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays)
def test_convert_data(data):
    assert np.allclose(data, dcst_private._convert_data(data), atol=atol)

    df = pd.DataFrame({'test': data})
    assert np.allclose(data, dcst_private._convert_data(df.loc[:,'test']),
                       atol=atol)
    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_data(df)
    excinfo.match('Input must be a 1D array or Pandas series.')

    s = pd.Series(data)
    assert np.allclose(data, dcst_private._convert_data(s), atol=atol)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.floats(-10, 10))
def test_convert_data_scalar(data):
    conv_data = dcst_private._convert_data(data)
    assert type(conv_data) == np.ndarray
    assert len(conv_data) == 1
    assert np.isclose(conv_data[0], data, atol=atol)


def test_convert_data_edge():
    x = np.array([1, 2, np.nan, 4])
    x_correct = np.array([1, 2, 4])
    x = dcst_private._convert_data(x)
    assert np.allclose(x, x_correct, atol=atol)

    x = np.array([1, 2, np.inf, 4])
    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_data(x, inf_ok=False)
    excinfo.match('All entries must be finite.')

    x_correct = np.array([1, 2, np.inf, 4])
    x = dcst_private._convert_data(x_correct, inf_ok=True)
    assert np.allclose(x, x_correct, atol=atol)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays_2)
def test_convert_two_data(data):
    x_correct, y_correct = data
    x, y = dcst_private._convert_two_data(x_correct, y_correct)
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)

    df = pd.DataFrame(data=data.transpose(), columns=['test1', 'test2'])
    x, y = dcst_private._convert_two_data(df['test1'], df['test2'])
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)

    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(df, df['test2'])
    excinfo.match('Input must be a 1D array or Pandas series.')

    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(df['test1'], df)
    excinfo.match('Input must be a 1D array or Pandas series.')

    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(df, df)
    excinfo.match('Input must be a 1D array or Pandas series.')


def test_convert_two_data_edge():
    x_correct = np.array([1, 2, 3])
    y_correct = np.array([4, 5, 6])
    x, y = dcst_private._convert_two_data(x_correct, y_correct)
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)

    x = np.array([1, 2, np.nan, 4])
    y = np.array([4, np.nan, 6, 7])
    x_correct, y_correct = np.array([1, 4]), np.array([4, 7])
    x, y = dcst_private._convert_two_data(x, y)
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)

    x = np.array([1, 2, np.inf, 4])
    y = np.array([4, 5, 6, 7])
    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(x, y, inf_ok=False)
    excinfo.match('All entries in arrays must be finite.')

    x_correct = np.array([1, 2, np.inf, 4])
    y_correct = np.array([4, 5, 6, 7])
    x, y = dcst_private._convert_two_data(x_correct, y_correct, inf_ok=True)
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)

    x = np.array([1, np.nan, np.nan, 4])
    y = np.array([np.nan, np.nan, 6, 7])
    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(x, y, min_len=2)
    excinfo.match('Arrays must have at least 2 mutual non-NaN entries.')

    x = np.array([1, np.nan, np.nan, 4])
    y = np.array([np.nan, np.nan, 6, 7])
    x, y = dcst_private._convert_two_data(x, y, min_len=1)
    x_correct = np.array([4])
    y_correct = np.array([7])
    assert np.allclose(x, x_correct, atol=atol)
    assert np.allclose(y, y_correct, atol=atol)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(0, 1000000))
def test_pandas_conversion(seed):
    df = pd.DataFrame({'a': [3, 2, 1, 4],
                       'b': [8, 6, 7, 5],
                       'c': [9.1, 10.1, 11.1, np.nan]})

    x, y = dcst.ecdf(df.loc[:, 'a'])
    assert (x == np.array([1, 2, 3, 4])).all()
    assert (y == np.array([0.25, 0.5, 0.75, 1.0])).all()

    x, y = dcst.ecdf(df.loc[:, 'c'])
    assert np.allclose(x, np.array([9.1, 10.1, 11.1]))
    assert np.allclose(y, np.array([1/3, 2/3, 1.0]))

    df = pd.DataFrame({
        'a': np.concatenate((np.random.normal(0, 1, size=10), [np.nan]*990)),
        'b': np.random.normal(0, 1, size=1000)})
    correct, _ = st.ks_2samp(df['a'].dropna(), df['b'])
    assert np.isclose(dcst.ks_stat(df['a'], df['b']), correct)

    df = pd.DataFrame({
        'a': np.concatenate((np.random.normal(0, 1, size=80), [np.nan]*20)),
        'b': np.random.normal(0, 1, size=100)})
    dcst_private._seed_numba(seed)
    correct = dcst.draw_bs_reps(df['a'].values, np.mean, size=100)
    dcst_private._seed_numba(seed)
    assert np.allclose(dcst.draw_bs_reps(df['a'], np.mean, size=100), correct,
                       atol=atol)

    dcst_private._seed_numba(seed)
    correct = dcst.draw_bs_reps(df['b'].values, np.mean, size=100)
    dcst_private._seed_numba(seed)
    assert np.allclose(dcst.draw_bs_reps(df['b'], np.mean, size=100), correct,
                       atol=atol)

    dcst_private._seed_numba(seed)
    correct = dcst.draw_perm_reps(df['a'].values, df['b'].values,
                                  dcst.diff_of_means, size=100)
    dcst_private._seed_numba(seed)
    assert np.allclose(dcst.draw_perm_reps(df['a'], df['b'],
                       dcst.diff_of_means, size=100), correct, atol=atol)
