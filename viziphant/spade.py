"""
Spike Pattern Detection and Evaluation (SPADE) plots
----------------------------------------------------

Visualizes output of :func:`elephant.spade.spade` analysis.

.. autosummary::
    :toctree: toctree/spade/

    plot_patterns_statistics
    plot_patterns

"""
# Copyright 2017-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.

from collections import defaultdict

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from viziphant.rasterplot import rasterplot


def plot_patterns_statistics(patterns):
    """
    This function creates a histogram plot to visualise patterns statistics
    output of a SPADE analysis.

    Parameters
    ----------
    patterns : list of dict
        The output of `elephant.spade.spade`.

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
        from elephant.spike_train_generation import homogeneous_poisson_process
        import viziphant
        np.random.seed(12)

        spiketrains = [homogeneous_poisson_process(rate=10 * pq.Hz,
                       t_stop=10 * pq.s) for _ in range(10)]
        patterns = spade.spade(spiketrains, bin_size=100*pq.ms,
                               winlen=1)['patterns']

        viziphant.spade.plot_patterns_statistics(patterns)
        plt.show()

    """
    stats = defaultdict(list)
    for pattern in patterns:
        stats['neurons'].append(pattern['neurons'])
        stats['occurrences'].append(len(pattern['times']))
        stats['pattern_size'].append(len(pattern['neurons']))
        stats['lags'].append(pattern['lags'])

    lags, lags_counts = np.unique(np.hstack(stats['lags']), return_counts=True)
    if len(lags) == 1:
        # case of only synchronous patterns
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    else:
        # case of patterns with delays
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        # adding panel with histogram of lags for delayed patterns
        axes[3].bar(lags, lags_counts)
        axes[3].set_xlabel('Lags (ms)')
        axes[3].set_ylabel('Count')
    plt.subplots_adjust(hspace=0.5)
    axes[0].set_title('Patterns statistics')

    neurons_participated, counts = np.unique(np.hstack(stats['neurons']),
                                             return_counts=True)
    axes[0].bar(neurons_participated, counts)
    axes[0].set_xlabel('Neuronal participation in patterns')
    axes[0].set_ylabel('Count')

    occurrences, counts = np.unique(stats['occurrences'],
                                    return_counts=True)
    axes[1].bar(occurrences, counts)
    axes[1].set_xlabel('Pattern occurrences')
    axes[1].set_ylabel('Count')

    sizes, counts = np.unique(stats['pattern_size'],
                              return_counts=True)
    axes[2].bar(sizes, counts)
    axes[2].set_xlabel('Pattern size')
    axes[2].set_ylabel('Count')

    return fig, axes


def plot_patterns(spiketrains, patterns, circle_sizes=(3, 50, 70)):
    """
    Simple plot showing a raster plot in gray along with one or more chosen
    SPADE patterns represented in color.

    Patterns that share neurons at a particular spike time are represented as
    pie charts of individual pattern colors.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of `neo.SpikeTrain` that were used as the input.
    patterns : dict or list of dict
        One or more patterns from a list of found patterns returned by
        :func:`elephant.spade.spade` function.
    circle_sizes : tuple of float, optional
        A tuple of 3 elements:
          1) raster plot neurons size that don't participate in the patterns;

          2) patterns circle size;

          3) pie chart (overlapped patterns) size.

        Default: (3, 50, 70)

    Returns
    -------
    axes : matplotlib.axes.Axes

    Examples
    --------
    In the plot below, two SPADE patterns are shown with pie chars representing
    the overlapping patterns, which happen to have at least one common neuron
    active at the same time across two patterns found by SPADE analysis,
    letting the user to explore pattern sizes (the number of neurons
    participating in a pattern) versus the number of occurrences.

    The bin size is set to a large value (``400ms``) in order to find
    "patterns" from 10 realizations of homogeneous Poisson process.

    .. plot::
        :include-source:

        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant import spade
        from elephant.spike_train_generation import homogeneous_poisson_process
        import viziphant
        np.random.seed(5)

        spiketrains = [homogeneous_poisson_process(rate=1 * pq.Hz,
                       t_stop=10 * pq.s) for _ in range(10)]
        patterns = spade.spade(spiketrains, bin_size=400 * pq.ms,
                               winlen=1)['patterns']

        viziphant.spade.plot_patterns(spiketrains, patterns[:2])
        plt.show()

    Additionally, one can add events to the returned plot:

    .. code-block:: python

        axes = viziphant.spade.plot_patterns(spiketrains, patterns[:2])
        event = neo.Event([0.5, 3.8] * pq.s, labels=['Trig ON', 'Trig OFF'])
        viziphant.events.add_event(axes, event=event)

    """
    if isinstance(patterns, dict):
        patterns = [patterns]
    axes = rasterplot(spiketrains, c='darkgray', s=circle_sizes[0])
    units = spiketrains[0].units
    time_scalar = units.rescale('ms').item()
    patterns_overlap = defaultdict(lambda: defaultdict(list))
    cmap = plt.cm.get_cmap("hsv", len(patterns) + 1)
    colors = np.array([cmap(i) for i in range(len(patterns))])

    for pattern_id, pattern in enumerate(patterns):
        times_ms = pattern['times'].rescale(pq.ms).magnitude.astype(int)
        for neuron in pattern['neurons']:
            for t in times_ms:
                patterns_overlap[neuron][t].append(pattern_id)
    for pattern_id, pattern in enumerate(patterns):
        times_s = pattern['times'].rescale(units).magnitude
        times_ms = (times_s * time_scalar).astype(int)
        for neuron in pattern['neurons']:
            mask_overlapped = np.zeros(len(times_ms), dtype=bool)
            for i, t in enumerate(times_ms):
                mask_overlapped[i] = len(patterns_overlap[neuron][t]) > 1
            times_no_overlap = times_s[~mask_overlapped]
            axes.scatter(times_no_overlap,
                         np.repeat(neuron, repeats=len(times_no_overlap)),
                         c=[colors[pattern_id]], s=circle_sizes[1])

    pie_chart_size = circle_sizes[2]
    for neuron in patterns_overlap:
        for t, pattern_ids in patterns_overlap[neuron].items():
            if len(pattern_ids) == 1:
                continue
            t_sec = t / time_scalar
            pie_angles = np.linspace(0, 2 * np.pi, num=len(pattern_ids) + 1)
            for theta1, theta2, wedge_color in zip(pie_angles[:-1],
                                                   pie_angles[1:],
                                                   colors[pattern_ids]):
                arc = np.linspace(theta1, theta2)
                x = np.r_[0, np.cos(arc)]
                y = np.r_[0, np.sin(arc)]
                axes.scatter([t_sec], [neuron], marker=np.c_[x, y],
                             s=pie_chart_size, c=[wedge_color])

    axes.set_ylabel('Neuron')
    return axes
