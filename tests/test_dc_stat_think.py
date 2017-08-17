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

@hypothesis.given(arrays, arrays)
def test_ecdf_formal(x, data):
    correct = np.searchsorted(np.sort(data), x, side='right') / len(data)
    assert np.isclose(dcst.ecdf_formal(x, data), correct).all()


@hypothesis.given(arrays)
def test_ecdf(data):
    x, y = dcst.ecdf(data)
    x_correct, y_correct = original.ecdf(data)
    assert np.isclose(x, x_correct).all() and np.isclose(y, y_correct).all()


def test_ecdf_formal_for_plotting():
    data = np.array([2, 1, 3])
    y_correct = np.array([0, 0, 1, 1, 2, 2, 3, 3]) / 3
    x_correct = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    x, y = dcst.ecdf(data, formal=True, min_x=0, max_x=4)
    assert np.isclose(x, x_correct).all()
    assert np.isclose(y, y_correct).all()

    data = np.array([1, 2, 2, 3])
    y_correct = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]) / 4
    x_correct = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 4])
    x, y = dcst.ecdf(data, formal=True, min_x=0, max_x=4)
    assert np.isclose(x, x_correct).all()
    assert np.isclose(y, y_correct).all()


@hypothesis.given(arrays, hs.integers(0, 1000000))
def test_bootstrap_replicate_1d(data, seed):
    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.mean)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.mean)
    assert (np.isnan(x) and np.isnan(x_correct)) or np.isclose(x, x_correct)

    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.median)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.median)
    assert (np.isnan(x) and np.isnan(x_correct)) or np.isclose(x, x_correct)

    np.random.seed(seed)
    x = dcst.bootstrap_replicate_1d(data, np.std)
    np.random.seed(seed)
    x_correct = original.bootstrap_replicate_1d(data[~np.isnan(data)], np.std)
    assert (np.isnan(x) and np.isnan(x_correct)) or np.isclose(x, x_correct)


def test_bootstrap_replicate_1d_nan():
    with pytest.raises(RuntimeError) as excinfo:
        dcst.bootstrap_replicate_1d(np.array([np.nan, np.nan]), np.mean)
    excinfo.match('Array must have at least 1 non-NaN entries.')


@hypothesis.given(arrays, hs.integers(0, 1000000), hs.integers(1, 100))
def test_draw_bs_reps(data, seed, size):
    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.mean, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.mean, size=size)
    assert np.isclose(x, x_correct).all()

    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.median, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.median,
                                      size=size)
    assert np.isclose(x, x_correct).all()

    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, np.std, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], np.std, size=size)
    assert np.isclose(x, x_correct).all()

    def my_fun(data):
        return np.dot(data, data)
    np.random.seed(seed)
    x = no_numba.draw_bs_reps(data, my_fun, size=size)
    np.random.seed(seed)
    x_correct = original.draw_bs_reps(data[~np.isnan(data)], my_fun, size=size)
    assert np.isclose(x, x_correct).all()


def test_draw_bs_pairs_linreg():
    for n in range(4, 10):
        for size in [1, 10, 100]:
            x = np.random.random(n)
            y = 1.5 * x + 3.0 + (np.random.random(n) - 0.5) * 0.1

            seed = np.random.randint(0, 100000)

            np.random.seed(seed)
            slope, intercept = no_numba.draw_bs_pairs_linreg(x, y, size=size)
            np.random.seed(seed)
            slope_correct, intercept_correct = \
                                original.draw_bs_pairs_linreg(x, y, size=size)
            assert np.isclose(slope, slope_correct).all()
            assert np.isclose(intercept, intercept_correct).all()


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
    assert np.isclose(bs_reps, bs_reps_correct).all()


@hypothesis.given(arrays_2)
def test_convert_two_data(data):
    x, y = data
    x_correct, y_correct = dcst_private._convert_two_data(x, y)
    assert np.isclose(x, x_correct).all and np.isclose(y, y_correct).all()


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


def test_convert_two_data_edge():
    x_correct = np.array([1, 2, 3])
    y_correct = np.array([4, 5, 6])
    x, y = dcst_private._convert_two_data(x_correct, y_correct)
    assert np.isclose(x, x_correct).all and np.isclose(y, y_correct).all()

    x = np.array([1, 2, np.nan, 4])
    y = np.array([4, np.nan, 6, 7])
    x_correct, y_correct = np.array([1, 4]), np.array([4, 7])
    x, y = dcst_private._convert_two_data(x, y)
    assert np.isclose(x, x_correct).all and np.isclose(y, y_correct).all()

    x = np.array([1, 2, np.inf, 4])
    y = np.array([4, 5, 6, 7])
    with pytest.raises(RuntimeError) as excinfo:
        dcst_private._convert_two_data(x, y, inf_ok=False)
    excinfo.match('All entries in arrays must be finite.')

    x_correct = np.array([1, 2, np.inf, 4])
    y_correct = np.array([4, 5, 6, 7])
    x, y = dcst_private._convert_two_data(x_correct, y_correct, inf_ok=True)
    assert np.isclose(x, x_correct).all and np.isclose(y, y_correct).all()

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
    assert np.isclose(x, x_correct).all and np.isclose(y, y_correct).all()



def test_against_original():
    for n in range(1, 100):
        data = np.random.random(n)
        x_orig, y_orig = original.ecdf(data)
        x, y = dcst.ecdf(data)
        assert np.isclose(x, x_orig).all() and np.isclose(y, y_orig).all()

    for _ in range(100):
        for n in range(2, 10):
            x = np.random.random(n)
            y = np.random.random(n)
            assert np.isclose(dcst.pearson_r(x, y), original.pearson_r(x, y))

    for n in range(1, 100):
        data_1 = np.random.random(n)
        data_2 = np.random.random(n)
        assert np.isclose(dcst.diff_of_means(data_1, data_2),
                          original.diff_of_means(data_1, data_2))


def test_permutation_sample():
    data_1 = np.random.random(100)
    data_2 = np.random.random(70)
    perm_1, perm_2 = dcst.permutation_sample(data_1, data_2)
    assert len(perm_1) == len(data_1) and len(data_2) == len(perm_2)
    assert ~np.isclose(data_1, perm_1).all()
    assert ~np.isclose(data_2, perm_2).all()
    assert np.isclose(np.sort(np.concatenate((data_1, data_2))),
                      np.sort(np.concatenate((perm_1, perm_2)))).all()


def test_draw_perm_reps():
    data_1 = np.ones(10)
    data_2 = np.ones(10)
    perm_reps = dcst.draw_perm_reps(data_1, data_2, dcst.diff_of_means,
                                    size=100)
    assert (perm_reps == np.zeros(100)).all()

    def diff_of_medians(data_1, data_2):
        return np.median(data_1) - np.median(data_2)
    data_1 = np.ones(10)
    data_2 = np.ones(10)
    perm_reps = dcst.draw_perm_reps(data_1, data_2, diff_of_medians, size=100)
    assert (perm_reps == np.zeros(100)).all()


def test_ks_stat():
    for n in range(3, 20):
        for mu in np.linspace(-1, 1, 10):
            for sigma in np.logspace(-1, 2, 10):
                data_1 = np.random.normal(mu, sigma, size=n)
                data_2 = np.random.normal(mu, sigma, size=1000)
                correct, _ = st.ks_2samp(data_1, data_2)
                assert np.isclose(dcst.ks_stat(data_1, data_2), correct)


def test_pandas_conversion():
    df = pd.DataFrame({'a': [3, 2, 1, 4],
                       'b': [8, 6, 7, 5],
                       'c': [9.1, 10.1, 11.1, np.nan]})

    x, y = dcst.ecdf(df.loc[:, 'a'])
    assert (x == np.array([1, 2, 3, 4])).all()
    assert (y == np.array([0.25, 0.5, 0.75, 1.0])).all()

    x, y = dcst.ecdf(df.loc[:, 'c'])
    assert np.isclose(x, np.array([9.1, 10.1, 11.1])).all()
    assert np.isclose(y, np.array([1/3, 2/3, 1.0])).all()

    df = pd.DataFrame({
        'a': np.concatenate((np.random.normal(0, 1, size=10), [np.nan]*990)),
        'b': np.random.normal(0, 1, size=1000)})
    correct, _ = st.ks_2samp(df['a'].dropna(), df['b'])
    assert np.isclose(dcst.ks_stat(df['a'], df['b']), correct)

    df = pd.DataFrame({
        'a': np.concatenate((np.random.normal(0, 1, size=80), [np.nan]*20)),
        'b': np.random.normal(0, 1, size=100)})
    dcst_private._seed_numba(42)
    correct = dcst.draw_bs_reps(df['a'].values, np.mean, size=100)
    dcst_private._seed_numba(42)
    assert np.isclose(dcst.draw_bs_reps(df['a'], np.mean, size=100), correct).all()

    dcst_private._seed_numba(42)
    correct = dcst.draw_bs_reps(df['b'].values, np.mean, size=100)
    dcst_private._seed_numba(42)
    assert np.isclose(dcst.draw_bs_reps(df['b'], np.mean, size=100), correct).all()

    dcst_private._seed_numba(42)
    correct = dcst.draw_perm_reps(df['a'].values, df['b'].values,
                                  dcst.diff_of_means, size=100)
    dcst_private._seed_numba(42)
    assert np.isclose(dcst.draw_perm_reps(df['a'], df['b'], dcst.diff_of_means,
                      size=100), correct).all()


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
    assert np.isclose(correct, result).all()




def test_pearson_r_nan():
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



def test_studentized_diff_of_means():
    data_1 = np.ones(10)
    data_2 = 2*np.ones(10)
    assert np.isnan(dcst.studentized_diff_of_means(data_1, data_2))

    data_1 = np.array([1, 2, 3, 4])
    data_2 = np.array([7, 5, 9, 6, 8])
    assert np.isclose(dcst.studentized_diff_of_means(data_1, data_2),
                      -4.7000967108038418)
