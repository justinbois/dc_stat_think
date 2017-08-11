#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `dc_stat_think` package."""

import numpy as np
import pandas as pd

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
