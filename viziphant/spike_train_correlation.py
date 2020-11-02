"""
Simple plotting function for spike train correlation measures
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.ticker import MaxNLocator


def plot_corrcoef(cc, corr_limits='auto', style='ticks', cmap='bwr',
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
    corr_limits : {'auto', 'full'} or list of float or tuple of float
        If list or tuple, the first element is the minimum and the second
        element is the maximum correlation for color mapping.
        If 'auto', the maximum absolute value of the non-diagonal coefficients
        will be used symmetrically for color mapping.
        If 'full', maximum correlation is set at 1.0 and mininum at -1.0.
        Default: 'auto'
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

    Raises
    ------
    ValueError
        If `corr_limits` is not one of: tuple, list, 'auto' or 'full'.
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

    # Get limits
    if isinstance(corr_limits, str):
        if corr_limits == 'full':
            vmin = -1.0
            vmax = 1.0
        elif corr_limits == 'auto':
            vmax = np.max(np.abs(np.triu(cc, k=1)))
            vmin = -vmax
        else:
            raise ValueError("Invalid limit specification. String must be"
                             "'full' or 'auto'.")
    elif isinstance(corr_limits, (list, tuple)):
        vmin, vmax = corr_limits
    else:
        raise ValueError("Invalid limit specification. Must be a list/tuple"
                         "of values or 'auto'/'full'.")

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
