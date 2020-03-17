"""
Simple plotting functions for statistical measures of spike trains
"""

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


def isi(isis, labels, binsize=2*pq.ms, cutoff=250*pq.ms):
    """
    This function creates a simple histogram plot to visualise an ISI distribution
    computed with elephant.statistics.isi.

    Multiple plots are created side by side for nested sublists in the isis and labels
    arguments.

    :param isis
        output of elephant.statistics.isi or list of multiple outputs
    :param labels
        list of labels corresponding to the isi distributions
    :param binsize
        binsize for the histogram. Default: 2 ms
    :param cutoff
        largest ISI to consider. Default: 250 ms
    """

    if hasattr(isis, 'units'):
        isis = [isis]
    if isinstance(labels, str):
        labels = [labels]

    fig, axes = plt.subplots(1, len(labels), sharex=True, sharey=True,
                             figsize=(8 * len(labels), 3))

    axes = np.atleast_1d(axes)

    bins = np.arange(0, cutoff.rescale(isis[0].units).magnitude.item(),
                     binsize.rescale(isis[0].units).magnitude.item())

    for idx, label in enumerate(labels):
        axes[idx].hist(isis[idx], bins=bins)
        axes[idx].set_title(labels[idx])
        axes[idx].set_xlabel(f'ISI Length ({isis[0].dimensionality.string})')
    axes[0].set_ylabel('ISI\nHistogram')

    return fig, axes

