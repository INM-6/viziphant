"""
Simple plotting function for spike train correlation measures
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.ticker import MaxNLocator


def plot_corrcoef(cc, vmin=-1, vmax=1, style='ticks', cmap='bwr',
                  cax_aspect=20, cax_pad_fraction=.5, figsize=(8, 8),
                  remove_diagonal=True):
    """
    This function plots the cross-correlation matrix returned by
    `elephant.spike_train_correlation.corrcoef` and adds a colour bar.

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
    style: {'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'} or dict, optional
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
        cc[np.diag_indices(cc.shape[0])] = 0

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
