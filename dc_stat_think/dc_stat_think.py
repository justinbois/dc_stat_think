# -*- coding: utf-8 -*-

"""Utilities for DataCamp's statistical thinking courses."""

import numpy as np
import numba

@numba.jit(nopython=True)
def ecdf_formal(x, data):
    output = np.empty_like(x)

    data = np.sort(data)

    for i, x_val in enumerate(x):
        j = 0
        while j < len(data) and x_val >= data[j]:
            j += 1

        output[i] = j

    return output / len(data)


@numba.jit(nopython=True)
def ecdf(data, formal=False, buff=0.1, min_x=None, max_x=None):
    """
    Generate `x` and `y` values for plotting an ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    formal : bool, default False
        If True, generate `x` and `y` values for formal ECDF.
        Otherwise, generate `x` and `y` values for "dot" style ECDF.
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a
        fraction of the total range of the data. Ignored if
        `formal` is False.
    min_x : float, default None
        Minimum value of `x` to include on plot. Overrides `buff`.
        Ignored if `formal` is False.
    max_x : float, default None
        Maximum value of `x` to include on plot. Overrides `buff`.
        Ignored if `formal` is False.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """

    if formal:
        return _ecdf_formal(data, buff=buff, min_x=min_x, max_x=max_x)
    else:
        return _ecdf_dots(data)


@numba.jit(nopython=True)
def _ecdf_dots(data):
    """
    Compute `x` and `y` values for plotting an ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """
    return np.sort(data), np.arange(1, len(data)+1) / len(data)


@numba.jit(nopython=True)
def _ecdf_formal(data, buff=0.1, min_x=None, max_x=None):
    """
    Generate `x` and `y` values for plotting a formal ECDF.

    Parameters
    ----------
    data : array_like
        Array of data to be plotted as an ECDF.
    buff : float, default 0.1
        How long the tails at y = 0 and y = 1 should extend as a fraction
        of the total range of the data.
    min_x : float, default None
        Minimum value of `x` to include on plot. Overrides `buff`.
    max_x : float, default None
        Maximum value of `x` to include on plot. Overrides `buff`.

    Returns
    -------
    x : array
        `x` values for plotting
    y : array
        `y` values for plotting
    """
    # Get x and y values for data points
    x, y = _ecdf_dots(data)

    # Set defaults for min and max tails
    if min_x is None:
        min_x = x[0] - (x[-1] - x[0])*buff
    if max_x is None:
        max_x = x[-1] + (x[-1] - x[0])*buff

    # Set up output arrays
    x_formal = np.empty(2*(len(x) + 1))
    y_formal = np.empty(2*(len(x) + 1))

    # y-values for steps
    y_formal[:2] = 0
    y_formal[2::2] = y
    y_formal[3::2] = y

    # x- values for steps
    x_formal[0] = min_x
    x_formal[1] = x[0]
    x_formal[2::2] = x
    x_formal[3:-1:2] = x[1:]
    x_formal[-1] = max_x

    return x_formal, y_formal


def eccdf(data):
    """Generate x and y values for plotting an ECCDF."""
    return np.sort(data), np.arange(len(data), 0, -1) / len(data)


def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])


@numba.jit(nopython=True)
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from
    inds = np.arange(len(x))

    # Initialize samples
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Take samples
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        A = np.vstack([bs_x, np.ones(len(bs_x))]).transpose()
        bs_slope_reps[i], bs_intercept_reps[i] = np.linalg.lstsq(A, bs_y)[0]

    return bs_slope_reps, bs_intercept_reps


def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for single statistic."""

    # Set up array of indices to sample from
    inds = np.arange(len(x))

    # Initialize replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates


@numba.jit(nopython=True)
def permutation_sample(data_1, data_2):
    permuted_data = np.random.permutation(np.concatenate((data_1, data_2)))
    return permuted_data[:len(data_1)], permuted_data[len(data_1):]


@numba.jit(nopython=True)
def draw_perm_reps(d1, d2, func, size=1):
    return np.array([func(*permutation_sample(d1, d2)) for i in range(size)])


@numba.jit(nopython=True)
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    return np.mean(data_1) - np.mean(data_2)


def pearson_r(data_1, data_2):
    return np.corrcoef(data_1, data_2)[0,1]


@numba.jit(nopython=True)
def ks_stat(data1, data2):
    # Compute ECDF from data
    x, y = ecdf(data1)

    # Compute corresponding values of the target CDF
    cdf = ecdf_formal(x, data2)

    # Compute distances between convex corners and CDF
    D_top = y - cdf

    # Compute distance between concave corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max(np.concatenate((D_top, D_bottom)))


def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):

    if f == np.random.exponential:
        return _draw_ks_reps_exp(n, args[0], size, n_reps)

    if f == np.random.normal:
        return _draw_ks_reps_norm(n, args[0], args[1], size, n_reps)

    # Generate samples from target distribution
    x_f = f(*args, size=size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        x_samp = f(*args, size=n)
        reps[i] = ks_stat(x_samp, x_f)
    return reps


@numba.jit(nopython=True)
def _draw_ks_reps_exp(n, x_mean, size, n_reps):
    # Generate samples from target distribution
    x_f = np.random.exponential(x_mean, size=size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        x_samp = np.random.exponential(x_mean, size=n)
        reps[i] = ks_stat(x_samp, x_f)
    return reps


@numba.jit(nopython=True)
def _draw_ks_reps_norm(n, x_mean, x_std, size, n_reps):
    # Generate samples from target distribution
    x_f = np.random.normal(x_mean, x_std, size=size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        x_samp = np.random.normal(x_mean, x_std, size=n)
        reps[i] = ks_stat(x_samp, x_f)
    return reps
