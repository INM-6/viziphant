"""
Simple plotting functions for statistical measures of spike trains
"""

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq


def plot_isi(intervals, label, binsize=2 * pq.ms, cutoff=250 * pq.ms):
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
