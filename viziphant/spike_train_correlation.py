"""
Simple plotting function for spike train correlation measures
"""

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


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
        cch, surr_cchs=None, significance_threshold=3.0,
        maxlag=None, figsize=None, legend=True, units=pq.s,
        title='Cross-correlation histogram',
        xlabel=None, ylabel=''):
    """
    Plot a cross correlation histogram, rescaled to seconds.

    Parameters
    ----------
    cch : neo.AnalogSignal
        A result of
        :func:`elephant.spike_train_correlation.cross_correlation_histogram`
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
        from elephant.spike_train_correlation import
             cross_correlation_histogram
        from viziphant.spike_train_correlation import
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
