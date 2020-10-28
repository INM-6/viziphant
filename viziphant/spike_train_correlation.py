"""
Simple plotting function for spike train correlation measures
"""
# Copyright 2019-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.


from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plot_corrcoef(corrcoef_matrix, axes, correlation_minimum=-1.,
                  correlation_maximum=1., colormap='bwr', color_bar_aspect=20,
                  color_bar_padding_fraction=.5, remove_diagonal=True):
    """
    Plots the cross-correlation matrix returned by
    :py:func:`elephant.spike_train_correlation.corrcoef` function and adds a
    color bar.

    Parameters
    ----------
    corrcoef_matrix : np.ndarray
        Pearson's correlation coefficient matrix
    axes : matplotlib.axes.Axes
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
    remove_diagonal : bool
        If True, the values in the main diagonal are replaced with zeros.
        Default: True

    Examples
    --------
    Create 10 homogeneous random Poisson spike trains of rate `10Hz` and bin
    the spikes into bins of `100ms` width, which is relatively large for such
    a firing rate, so we expect non-zero correlations.

    >>> import quantities as pq
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> from elephant.conversion import BinnedSpikeTrain
    >>> from elephant.spike_train_correlation import correlation_coefficient
    >>> np.random.seed(0)
    >>> spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
    ...                t_stop=10*pq.s) for _ in range(10)]
    >>> binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=100*pq.ms)
    >>> corrcoef_matrix = correlation_coefficient(binned_spiketrains)
    >>> fig, axes = plt.subplots()
    >>> plot_corrcoef(corrcoef_matrix, axes=axes)
    >>> axes.set_xlabel('Neuron')
    >>> axes.set_ylabel('Neuron')
    >>> axes.set_title("Correlation coefficient matrix")
    """
    if remove_diagonal:
        corrcoef_matrix = corrcoef_matrix.copy()
        np.fill_diagonal(corrcoef_matrix, val=0)

    image = axes.imshow(corrcoef_matrix,
                        vmin=correlation_minimum, vmax=correlation_maximum,
                        cmap=colormap)

    # Initialise colour bar axis
    divider = make_axes_locatable(axes)
    width = axes_size.AxesY(axes, aspect=1. / color_bar_aspect)
    pad = axes_size.Fraction(color_bar_padding_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)

    plt.colorbar(image, cax=cax)
