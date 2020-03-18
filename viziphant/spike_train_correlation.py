"""
Simple plotting function for spike train correlation measures
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.ticker import MaxNLocator


def plot_corrcoef(cc, vmin=-1, vmax=1, style='ticks', cmap='bwr',
             cax_aspect=20, cax_pad_fraction=.5, figsize=(8, 8)):
    """
    This function plots the cross-correlation matrix returned by
    elephant.spike_train_correlation.corrcoef and adds a colour bar.

    Parameters
    ----------
    cc : np.ndarray
        The output of elephant.spike_train_correlation.corrcoef.
    vmin : int or float
        The minimum correlation for colour mapping. Default: -1
    vmax : int or float
        The maximum correlation for colour mapping. Default: 1
    style: str
        A seaborn style setting. Default: 'ticks'
    cmap : str
        The colour map. Default: 'bwr'
    cax_aspect : int or float
        The aspect ratio of the colour bar. Default: 20
    cax_pad_fraction : int or float
        The padding between matrix plot and colour bar
        relative to colour bar width. Default: .5
    figsize : tuple of int
        The size of the figure. Default (8, 8)

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
