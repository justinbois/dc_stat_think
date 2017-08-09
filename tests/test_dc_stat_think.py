#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `dc_stat_think` package."""

import numpy as np

import pytest

import dc_stat_think as dcst

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
        for n in range(1, 10):
            x = np.random.random(n)
            y = np.random.random(n)
            assert np.isclose(dcst.pearson_r(x, y), np.corrcoef(x, y)[0,1])

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
