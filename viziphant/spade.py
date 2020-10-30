"""
Spike Pattern Detection and Evaluation (SPADE) plots
----------------------------------------------------

Visualizes output of :func:`elephant.spade.spade` analysis.

.. autosummary::
    :toctree: toctree/spade/

    plot_patterns_statistics
    plot_pattern

"""
# Copyright 2017-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.

from collections import defaultdict

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from viziphant.rasterplot import plot_raster


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

    Examples
    --------
    .. plot::
        :include-source:

        import neo
        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant import spade
        from viziphant.spade import plot_patterns_statistics
        np.random.seed(12)

        spiketrains = [neo.SpikeTrain((np.arange(20)+np.random.rand(20))*pq.s,
                       t_stop=21) for _ in range(50)]
        patterns = spade.spade(spiketrains, bin_size=100*pq.ms,
                               winlen=1)['patterns']

        plot_patterns_statistics(patterns, winlen=1, bin_size=100*pq.ms,
                                 n_neurons=len(spiketrains))
        plt.show()

    """
    patterns_dict = defaultdict(list)
    for pattern in patterns:
        patterns_dict['neurons'].append(pattern['neurons'])
        patterns_dict['occurrences'].append(len(pattern['times']))
        patterns_dict['pattern_size'].append(len(pattern['neurons']))
        patterns_dict['lags'].append(pattern['lags'])
    patterns_dict['neurons'] = np.hstack(patterns_dict['neurons'])
    patterns_dict['lags'] = np.hstack(patterns_dict['lags'])
    if winlen == 1:
        # case of only synchronous patterns
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    else:
        # case of patterns with delays
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    axes[0].set_title('Patterns statistics')
    axes[0].hist(patterns_dict['neurons'], bins=n_neurons)
    axes[0].set_xlabel('Neuronal participation in patterns')
    axes[0].set_ylabel('Count')
    occurrences, counts = np.unique(patterns_dict['occurrences'],
                                    return_counts=True)
    axes[1].bar(occurrences, counts)
    axes[1].set_xlabel('Pattern occurrences')
    axes[1].set_ylabel('Count')
    sizes, counts = np.unique(patterns_dict['pattern_size'],
                              return_counts=True)
    axes[2].bar(sizes, counts)
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


def plot_pattern(spiketrains, pattern):
    """
    Simple plot showing a rasterplot in gray along with one chosen SPADE
    pattern with its spikes represented in red.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of `neo.SpikeTrain` that were used as the input.
    pattern : dict
        A pattern from a list of found patterns returned by
        :func:`elephant.spade.spade` function.

    Returns
    -------
    axes : matplotlib.axes.Axes

    Examples
    --------
    .. plot::
        :include-source:

        import neo
        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant import spade
        from viziphant.spade import plot_pattern
        np.random.seed(12)

        spiketrains = [neo.SpikeTrain(
            np.sort((np.arange(20) + 3 * np.random.rand(20))) * pq.s,
            t_stop=23) for _ in range(10)
        ]
        patterns = spade.spade(spiketrains, bin_size=500*pq.ms,
                               winlen=1)['patterns']

        plot_pattern(spiketrains, pattern=patterns[0])
        plt.show()

    """
    axes = plot_raster(spiketrains, color='darkgray', ms=3)
    pattern_times = pattern['times'].rescale(pq.s).magnitude
    # for each neuron that participated in the pattern
    for neuron in pattern['neurons']:
        axes.plot(pattern_times, [neuron] * len(pattern_times),
                  'o', color='red', ms=7)
    axes.set_ylabel('Neuron')
    return axes
