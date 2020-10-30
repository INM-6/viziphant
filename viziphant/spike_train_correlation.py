"""
Spike train correlation plots
-----------------------------

.. autosummary::
    :toctree: toctree/spike_train_correlation/

    plot_corrcoef
    plot_cross_correlation_histogram

"""
# Copyright 2019-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.


from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plot_corrcoef(corrcoef_matrix, axes, correlation_minimum=-1.,
                  correlation_maximum=1., colormap='bwr', color_bar_aspect=20,
                  color_bar_padding_fraction=.5, remove_diagonal=True):
    """
    Plots a cross-correlation matrix returned by
    :func:`elephant.spike_train_correlation.correlation_coefficient`
    function and adds a color bar.

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

    .. plot::
       :include-source:

        import quantities as pq
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant.conversion import BinnedSpikeTrain
        from elephant.spike_train_correlation import correlation_coefficient
        from viziphant.spike_train_correlation import plot_corrcoef
        np.random.seed(0)

        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=100*pq.ms)
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        fig, axes = plt.subplots()
        plot_corrcoef(corrcoef_matrix, axes=axes)
        axes.set_xlabel('Neuron')
        axes.set_ylabel('Neuron')
        axes.set_title("Correlation coefficient matrix")
        plt.show()

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


def plot_cross_correlation_histogram(
        cch, surr_cchs=None, significance_threshold=3.0,
        maxlag=None, figsize=None, legend=True, units=pq.s,
        title='Cross-correlation histogram',
        xlabel=None, ylabel=''):
    """
    Plot a cross-correlation histogram returned by
    :func:`elephant.spike_train_correlation.cross_correlation_histogram`,
    rescaled to seconds.

    Parameters
    ----------
    cch : neo.AnalogSignal
        Cross-correlation histogram.
    surr_cchs : np.ndarray or neo.AnalogSignal, optional
        Contains cross-correlation histograms for each surrogate realization.
        If None, only the original `cch` is plotted.
        Default: None
    significance_threshold : float or None, optional
        Number of standard deviations for significance threshold. If None,
        don't plot the standard deviation.
        Default: 3.0
    maxlag : pq.Quantity or None, optional
        Left and right borders of the plot.
        Default: None
    figsize : tuple or None, optional
        Figure size
        Default: None
    legend : bool, optional
        Whether to include the legend.
        Default: True
    units : pq.Quantity, optional
        Unit in which to the CCH time lag
        Default: pq.ms
    title : str, optional
        The plot title.
        Default: 'Cross-correlation histogram'
    xlabel : str or None, optional
        Label X axis. If None, it'll be set to `'Time lag (units)'`.
        Default: None
    ylabel : str, optional
        Label Y axis.
        Default: ''

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant.conversion import BinnedSpikeTrain
        from elephant.spike_train_correlation import \
             cross_correlation_histogram
        from viziphant.spike_train_correlation import \
             plot_cross_correlation_histogram

        spiketrain1 = homogeneous_poisson_process(rate=10*pq.Hz,
                                                  t_stop=10*pq.s)
        spiketrain2 = homogeneous_poisson_process(rate=10*pq.Hz,
                                                  t_stop=10*pq.s)
        binned_spiketrain1 = BinnedSpikeTrain(spiketrain1, bin_size=100*pq.ms)
        binned_spiketrain2 = BinnedSpikeTrain(spiketrain2, bin_size=100*pq.ms)
        cch, lags = cross_correlation_histogram(binned_spiketrain1,
                                                binned_spiketrain2)

        plot_cross_correlation_histogram(cch)
        plt.show()

    """
    fig, ax = plt.subplots(figsize=figsize)
    # plot the CCH of the original data
    cch_times = cch.times.rescale(units).magnitude
    ax.plot(cch_times, cch.magnitude, color='C0',
            label='raw CCH')

    if surr_cchs is not None:
        # Compute the mean CCH
        cch_mean = surr_cchs.mean(axis=0)

        # Plot the average from surrogates
        ax.plot(cch_times, cch_mean, lw=2, color='C2',
                label='mean surr. CCH')

        # compute the standard deviation and plot the significance threshold
        if significance_threshold is not None:
            cch_threshold = cch_mean + significance_threshold * surr_cchs.std(
                axis=0, ddof=1)

            ax.plot(cch_times, cch_threshold, lw=2, color='C3',
                    label='significance threshold')

    ax.set_title(title)
    if xlabel is None:
        xlabel = f"Time lag ({units.dimensionality})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if maxlag is not None:
        maxlag = maxlag.rescale(units).magnitude
        ax.set_xlim(-maxlag, maxlag)
    if legend:
        ax.legend()
    return fig, ax
