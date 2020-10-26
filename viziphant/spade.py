"""
Simple plotting functions for statistical measures of spike trains
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import neo
from viziphant.rasterplot import plot_raster


def plot_patterns_statistics(patterns, winlen, bin_size, n_neurons):
    """
    This function creates a histogram plot to visualise patterns statistics
    output of a SPADE analysis (:func:`elephant.spade.spade`).

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
    patterns_dict = defaultdict(list)
    for pattern in patterns:
        patterns_dict['neurons'].append(pattern['neurons'])
        patterns_dict['occurrences'].append(len(pattern['times']))
        patterns_dict['pattern_size'].append(len(pattern['neurons']))
        patterns_dict['lags'].append(pattern['lags'])
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
                 bins=np.arange(1.5, np.max(
                     patterns_dict['occurrences']) + 1, 1))
    axes[1].set_xlabel('Pattern occurrences')
    axes[1].set_ylabel('Count')
    axes[2].hist(patterns_dict['pattern_size'],
                 bins=np.arange(1.5, np.max(patterns_dict['pattern_size']), 1))
    axes[2].set_xlabel('Pattern size')
    axes[2].set_ylabel('Count')
    if winlen != 1:
        # adding panel with histogram of lags for delayed patterns
        axes[3].hist(patterns_dict['lags'],
                     bins=np.arange(-bin_size.magnitude / 2,
                                    winlen * bin_size.magnitude +
                                    bin_size.magnitude / 2,
                                    bin_size.magnitude))
        axes[3].set_xlabel('lags (ms)')
        axes3_xmax = winlen * bin_size.magnitude - bin_size.magnitude / 2
        axes[3].set_xlim([-bin_size.magnitude / 2, axes3_xmax])
        axes[3].set_ylabel('Count')
    return fig, axes


def _build_mask(spiketrains, pattern):
    mask_patterns = []
    for index, st in enumerate(spiketrains):
        if index in pattern['neurons']:
            patt_st = neo.SpikeTrain(pattern['times'],
                                     t_start=st.t_start,
                                     t_stop=st.t_stop)
        else:
            patt_st = neo.SpikeTrain([]*pq.ms,
                                     t_start=st.t_start,
                                     t_stop=st.t_stop)
        mask_patterns.append(patt_st)
    return mask_patterns


def plot_pattern(spiketrains, pattern):
    """
    Simple plot showing a rasterplot along with one chosen pattern with its
    spikes represented in red

    Parameters:
    spiketrains: list
        List of neo.SpikeTrain of the original data
    pattern : dictionary
        One pattern output of elephant.spade to be plotted

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = plot_raster(spiketrains)
    mask_patterns = _build_mask(spiketrains, pattern)
    for st_index, st in enumerate(mask_patterns):
        # Dot display
        if st.__len__():
            for patt_occurrence in st:
                ax.plot(patt_occurrence.rescale(pq.s).magnitude,
                        st_index,
                        '.',
                        color='red')
    ax.set_ylabel('neurons')
    return ax

