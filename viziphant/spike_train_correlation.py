"""
Simple plotting function for spike train correlation measures
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.ticker import MaxNLocator
import elephant.spike_train_correlation as corr
import elephant.conversion as conv


def plot_corrcoef(cc, vmin=-1, vmax=1, style='ticks', cmap='bwr',
                  cax_aspect=20, cax_pad_fraction=.5, figsize=(8, 8),
                  remove_diagonal=True):
    """
    This function plots the cross-correlation matrix returned by
    `elephant.spike_train_correlation.correlation_coefficient` and adds a
    colour bar.

    Parameters
    ----------
    cc : np.ndarray
        The output of
        `elephant.spike_train_correlation.correlation_coefficient`.
    vmin : int or float, optional
        The minimum correlation for colour mapping.
        Default: -1
    vmax : int or float, optional
        The maximum correlation for colour mapping.
        Default: 1
    style: {'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'} or dict,
           optional
        A seaborn style setting.
        Default: 'ticks'
    cmap : str, optional
        The colour map.
        Default: 'bwr'
    cax_aspect : int or float, optional
        The aspect ratio of the colour bar.
        Default: 20
    cax_pad_fraction : int or float, optional
        The padding between matrix plot and colour bar relative to colour bar
        width.
        Default: .5
    figsize : tuple of int, optional
        The size of the figure.
        Default: (8, 8)
    remove_diagonal : bool
        If True, the values in the main diagonal are replaced with zeros.
        Default: True

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    # Initialise plotting canvas
    sns.set_style(style)

    # Initialise figure and image axis
    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'},
                           figsize=figsize)

    # Remove the diagonal
    if remove_diagonal:
        cc = cc.copy()
        np.fill_diagonal(cc, val=0)

    im = ax.imshow(cc, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Initialise colour bar axis
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./cax_aspect)
    pad = axes_size.Fraction(cax_pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)

    plt.colorbar(im, cax=cax)

    return fig, ax


def plot_cross_correlation_histogram(
        cch, surr_cchs=None, significance_threshold=3.,
        maxlag=None, figsize=None,
        legend=True, title='', xlabel='', ylabel=''):
    """

    Parameters
    ----------
    cch : neo.AnalogSignal
        as a result of
        elephant.spike_train_correlation.cross_correlation_histogram()
    surr_cchs : np.ndarray of neo.AnalogSignal
        contains cchs for each surrogate realization
        If None, only the original cch is plotted
        Default : None
    significance_threshold : float
        Number of standard deviations for significance threshold
        Default : None
    maxlag : pq.Quantity
        left and right border of plot
        Default : None
    figsize : tuple
        figure size
        Default : None
    legend : bool
        Whether to include a legend
        Default : True
    title : str
        title of the plot
        Default : ''
    xlabel : str
        label of x-axis
        Default : ''
    ylabel : str
        label of y-axis
        Default : ''

    Returns
    -------
    fig, ax : figure, axis
    """

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    # plot the CCH of the original data
    plt.plot(cch.times.magnitude, cch.magnitude, color='C0',
             label='raw CCH')

    if surr_cchs is not None:
        # Compute the mean CCH
        cch_mean = surr_cchs.mean(axis=0)

        # Plot the average from surrogates
        plt.plot(cch.times.magnitude, cch_mean, lw=2, color='C2',
                 label='mean surr. CCH')

        # compute the standard deviation and plot the significance threshold
        if significance_threshold is not None:
            cch_threshold = cch_mean + 3 * surr_cchs.std(axis=0, ddof=1)

            plt.plot(cch.times.magnitude, cch_threshold, lw=2, color='C3',
                     label='significance threshold')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if maxlag is not None:
        maxlag.rescale(cch.times.units)
        plt.xlim(-maxlag, maxlag)
    if legend:
        plt.legend()
    return fig, ax
