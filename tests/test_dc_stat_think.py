#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `dc_stat_think` package."""

import numpy as np
import pandas as pd
import scipy.stats as st

import pytest

import dc_stat_think as dcst
import dc_stat_think.original as original

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

def test_draw_bs_reps():
    data = np.ones(10)
    assert (dcst.draw_bs_reps(data, np.mean, size=100) == np.ones(100)).all()

    data = np.ones(10)
    assert (dcst.draw_bs_reps(data, np.median, size=100) == np.ones(100)).all()

    data = np.ones(10)
    assert (dcst.draw_bs_reps(data, np.std, size=100) == np.zeros(100)).all()

    data = np.ones(10)
    assert (dcst.draw_bs_reps(data, np.max, size=100) == np.ones(100)).all()


def test_draw_bs_pairs_linreg():
    x = np.arange(100)
    y = 5 + 2 * x
    slopes, intercepts = dcst.draw_bs_pairs_linreg(x, y, size=10)
    assert np.isclose(slopes, 2).all() and np.isclose(intercepts, 5).all()


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
    dcst.seed_numba(42)
    correct = dcst.draw_bs_reps(df['a'].values, np.mean, size=100)
    dcst.seed_numba(42)
    assert np.isclose(dcst.draw_bs_reps(df['a'], np.mean, size=100), correct).all()

    dcst.seed_numba(42)
    correct = dcst.draw_bs_reps(df['b'].values, np.mean, size=100)
    dcst.seed_numba(42)
    assert np.isclose(dcst.draw_bs_reps(df['b'], np.mean, size=100), correct).all()

    dcst.seed_numba(42)
    correct = dcst.draw_perm_reps(df['a'].values, df['b'].values,
                                  dcst.diff_of_means, size=100)
    dcst.seed_numba(42)
    assert np.isclose(dcst.draw_perm_reps(df['a'], df['b'], dcst.diff_of_means,
                      size=100), correct).all()


def test_ecdf_formal():
    assert dcst.ecdf_formal(0.1, [0, 1, 2, 3]) == 0.25
    assert dcst.ecdf_formal(-0.1, [0, 1, 2, 3]) == 0.0
    assert dcst.ecdf_formal(0.1, [3, 2, 0, 1]) == 0.25
    assert dcst.ecdf_formal(-0.1, [3, 2, 0, 1]) == 0.0
    assert dcst.ecdf_formal(2, [3, 2, 0, 1]) == 0.75
    assert dcst.ecdf_formal(1, [3, 2, 0, 1]) == 0.5
    assert dcst.ecdf_formal(3, [3, 2, 0, 1]) == 1.0
    assert dcst.ecdf_formal(0, [3, 2, 0, 1]) == 0.25

    correct = np.array([np.nan, 1.0])
    result = dcst.ecdf_formal([np.nan, np.inf], [0, 1, 2, 3])
    assert np.isnan(result[0])
    assert result[1] == 1.0

    correct = np.array([np.nan, 1.0])
    result = dcst.ecdf_formal([np.nan, np.inf], [3, 2, 0, 1])
    assert np.isnan(result[0])
    assert result[1] == 1.0


def test_pearson_r():
    for _ in range(100):
        for n in range(2, 10):
            x = np.random.random(n)
            y = np.random.random(n)
            assert np.isclose(dcst.pearson_r(x, y), np.corrcoef(x, y)[0,1])


def test_studentized_diff_of_means():
    data_1 = np.ones(10)
    data_2 = 2*np.ones(10)
    assert np.isnan(dcst.studentized_diff_of_means(data_1, data_2))

    data_1 = np.array([1, 2, 3, 4])
    data_2 = np.array([7, 5, 9, 6, 8])
    assert np.isclose(dcst.studentized_diff_of_means(data_1, data_2),
                      -4.7000967108038418)
