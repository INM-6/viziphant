"""
Simple plotting functions for statistical measures of spike trains
"""

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


def plot_isi(intervals, label, binsize=2*pq.ms, cutoff=250*pq.ms):
    """
    This function creates a simple histogram plot to visualise an inter-spike
    interval (ISI) distribution computed with elephant.statistics.isi.

    Parameters
    ----------
    intervals : pq.Quantity
        The output of elephant.statistics.isi
    label : str
        The label of the ISI distribution
    binsize : pq.Quantity
        The bin size for the histogram. Default: 2 ms
    cutoff : pq.Quantity
        The largest ISI to consider. Default: 250 ms

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=(8, 3))

    bins = np.arange(0, cutoff.rescale(intervals.units).magnitude.item(),
                     binsize.rescale(intervals.units).magnitude.item())

    ax.hist(intervals, bins=bins)
    ax.set_title(f'{label} ISI distribution')
    ax.set_xlabel(f'Inter-spike interval ({intervals.dimensionality.string})')
    ax.set_ylabel('Count')

    return fig, ax


def plot_patterns_statistics(patterns, winlen, bin_size, n_neurons):
    """
    This function creates a histogram plot to visualise patterns statistics
    output of a SPADE analysis.

    Parameters
    ----------
    patterns : list
        The output of elephant.spade
    winlen : int
        window length of the SPADE analysis
    bin_size : pq.Quantity
        The bin size of the SPADE analysis
    n_neurons : int
        Number of neurons in the data set being analyzed

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    patterns_dict = {'neurons': np.array([]),
                     'occurrences': np.array([]),
                     'pattern_size': np.array([]),
                     'lags': np.array([])}
    for pattern in patterns:
        patterns_dict['neurons'] = \
            np.append(patterns_dict['neurons'], pattern['neurons'])
        patterns_dict['occurrences'] = \
            np.append(patterns_dict['occurrences'], len(pattern['times']))
        patterns_dict['pattern_size'] = \
            np.append(patterns_dict['pattern_size'], len(pattern['neurons']))
        patterns_dict['lags'] = \
            np.append(patterns_dict['lags'], pattern['lags'])
    if winlen == 1:
        # case of only synchronous patterns
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    else:
        # case of patterns with delays
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    axes[0].set_title('Patterns statistics')
    axes[0].hist(patterns_dict['neurons'],
                 bins=np.arange(0, n_neurons + 1))
    axes[0].set_xlabel('Neuronal participation in patterns')
    axes[0].set_ylabel('Count')
    axes[1].hist(patterns_dict['occurrences'],
                 bins=np.arange(1.5,
                                np.max(patterns_dict['occurrences']) + 1, 1))
    axes[1].set_xlabel('Pattern occurrences')
    axes[1].set_ylabel('Count')
    axes[2].hist(patterns_dict['pattern_size'],
                 bins=np.arange(1.5,
                 np.max(patterns_dict['pattern_size']), 1))
    axes[2].set_xlabel('Pattern size')
    axes[2].set_ylabel('Count')
    if winlen != 1:
        # adding panel with histogram of lags for delayed patterns
        axes[3].hist(patterns_dict['lags'],
                     bins=np.arange(-bin_size.magnitude/2,
                                    winlen*bin_size.magnitude +
                                    bin_size.magnitude/2,
                                    bin_size.magnitude))
        axes[3].set_xlabel('lags (ms)')
        axes[3].set_xlim([-bin_size.magnitude/2,
                          winlen*bin_size.magnitude - bin_size.magnitude/2])
        axes[3].set_ylabel('Count')
    return fig, axes
