"""
Simple plotting function for spike train correlation measures
"""
# Copyright 2019-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.


from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plot_corrcoef(
        correlation_coefficient_matrix, axes, correlation_minimum=-1.,
        correlation_maximum=1., colormap='bwr', color_bar_aspect=20,
        color_bar_padding_fraction=.5):

    """
    Plots the cross-correlation matrix returned by
    :py:func:`elephant.spike_train_correlation.corrcoef` function and adds a
    color bar.

    Parameters
    ----------
    correlation_coefficient_matrix : np.ndarray
        Pearson's correlation coefficient matrix
    axes : object
        Matplotlib figure Axes
    correlation_minimum : float
        minimum correlation for colour mapping. Default: -1
    correlation_maximum : float
        maximum correlation for colour mapping. Default: 1
    colormap : str
        colormap. Default: 'bwr'
    color_bar_aspect : float
        aspect ratio of the color bar. Default: 20
    color_bar_padding_fraction : float
        padding between matrix plot and color bar relative to color bar width.
        Default: .5

    Examples
    --------
    Create correlation coefficient matrix from Elephant `corrcoef` example
    and save the result to `corrcoef_matrix`.

    >>> import seaborn
    >>> seaborn.set_style('ticks')
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ...
    >>> plot_corrcoef(correlation_coefficient_matrix, axes=ax)

    """

    image = axes.imshow(correlation_coefficient_matrix,
                        vmin=correlation_minimum, vmax=correlation_maximum,
                        cmap=colormap)

    # Initialise colour bar axis
    divider = make_axes_locatable(axes)
    width = axes_size.AxesY(axes, aspect=1. / color_bar_aspect)
    pad = axes_size.Fraction(color_bar_padding_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)

    plt.colorbar(image, cax=cax)
