"""
Spike patterns plots
--------------------

Visualizes detected spike patterns returned by :func:`elephant.spade.spade`
or :func:`elephant.cell_assembly_detection.cell_assembly_detection` functions.

.. autosummary::
    :toctree: toctree/patterns/

    plot_patterns


Spike patterns statistics plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: toctree/patterns/

    plot_patterns_statistics_all
    plot_patterns_statistics_participation
    plot_patterns_statistics_occurrence
    plot_patterns_statistics_size
    plot_patterns_statistics_lags

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.

from collections import defaultdict

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from viziphant.rasterplot import rasterplot


def plot_patterns_statistics_participation(patterns, axes=None):
    """
    Create a histogram of neural participation in patterns.

    Parameters
    ----------
    patterns : list of dict
        The output of `elephant.spade.spade`.
    axes : matplotlib.axes.Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    See Also
    --------
    plot_patterns_statistics_all

    """
    if axes is None:
        fig, axes = plt.subplots()

    neurons = []
    for pattern in patterns:
        neurons.append(pattern['neurons'])
    neurons_participated, counts = np.unique(np.hstack(neurons),
                                             return_counts=True)

    axes.bar(neurons_participated, counts)
    axes.set_xlabel('Neuronal participation in patterns')
    axes.set_ylabel('Count')

    return axes


def plot_patterns_statistics_occurrence(patterns, axes=None):
    """
    Create a histogram of pattern occurrences.

    Parameters
    ----------
    patterns : list of dict
        The output of `elephant.spade.spade`.
    axes : matplotlib.axes.Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    See Also
    --------
    plot_patterns_statistics_all

    """
    if axes is None:
        fig, axes = plt.subplots()

    occurrence = []
    for pattern in patterns:
        occurrence.append(len(pattern['times']))
    occurrences, counts = np.unique(occurrence, return_counts=True)

    axes.bar(occurrences, counts)
    axes.set_xlabel('Pattern occurrences')
    axes.set_ylabel('Count')

    return axes


def plot_patterns_statistics_size(patterns, axes=None):
    """
    Create a histogram of pattern sizes.

    Parameters
    ----------
    patterns : list of dict
        The output of `elephant.spade.spade`.
    axes : matplotlib.axes.Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    See Also
    --------
    plot_patterns_statistics_all

    """
    if axes is None:
        fig, axes = plt.subplots()

    size = []
    for pattern in patterns:
        size.append(len(pattern['neurons']))
    size_unique, counts = np.unique(size, return_counts=True)

    axes.bar(size_unique, counts)
    axes.set_xlabel('Pattern size')
    axes.set_ylabel('Count')

    return axes


def plot_patterns_statistics_lags(patterns, axes=None):
    """
    Create a histogram of pattern lags.

    Parameters
    ----------
    patterns : list of dict
        The output of `elephant.spade.spade`.
    axes : matplotlib.axes.Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    See Also
    --------
    plot_patterns_statistics_all

    """
    if axes is None:
        fig, axes = plt.subplots()
    
    # 'times' and 'lags' share the same units;
    # however, only lag units are of interest
    units = patterns[0]['lags'].units
    lags = []
    for pattern in patterns:
        lags.append(pattern['lags'].magnitude)
    lags_unique, counts = np.unique(np.hstack(lags), return_counts=True)

    axes.bar(lags_unique, counts)
    axes.set_xlabel(f'Lags ({units.dimensionality})')
    axes.set_ylabel('Count')

    return axes


def plot_patterns_statistics_all(patterns):
    """
    Create a histogram plot to visualise all patterns statistics of SPADE
    analysis output.

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

        viziphant.patterns.plot_patterns_statistics_all(patterns)
        plt.show()

    """
    lags = []
    for pattern in patterns:
        lags.append(pattern['lags'].magnitude)
    lags_unique, counts = np.unique(np.hstack(lags), return_counts=True)
    if len(lags_unique) == 1:
        # case of only synchronous patterns
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    else:
        # case of patterns with delays
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        # adding panel with histogram of lags for delayed patterns
        plot_patterns_statistics_lags(patterns, axes=axes[3])
    plt.suptitle('Patterns statistics')
    plot_patterns_statistics_participation(patterns, axes=axes[0])
    plot_patterns_statistics_occurrence(patterns, axes=axes[1])
    plot_patterns_statistics_size(patterns, axes=axes[2])
    plt.subplots_adjust(hspace=0.5)

    return axes


def plot_patterns(spiketrains, patterns, circle_sizes=(3, 50, 70),
                  colors=None):
    """
    Raster plot with one or more chosen SPADE or CAD patterns ot top shown in
    color.

    Overlapping patterns (patterns that share neurons at a particular spike
    time) are represented as pie charts of individual pattern colors.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of spike trains that were used as the input.
    patterns : dict or list of dict
        One or more patterns from a list of found patterns returned by
        :func:`elephant.spade.spade` or
        :func:`elephant.cell_assembly_detection.cell_assembly_detection`
        pattern detectors.
    circle_sizes : tuple of float, optional
        A tuple of 3 elements:
          1) raster plot neurons size that don't participate in the patterns;

          2) patterns circle size;

          3) pie chart (overlapped patterns) size.

        Default: (3, 50, 70)
    colors : list of str or None
        A user-defined list of pattern colors. If None, the HSV colormap will
        be used to pick a different color for each pattern.
        Default: None

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

        axes = viziphant.patterns.plot_patterns(spiketrains, patterns[:2])
        plt.show()

    CAD example:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        from elephant.cell_assembly_detection import cell_assembly_detection
        from elephant.conversion import BinnedSpikeTrain
        from elephant.spike_train_generation import compound_poisson_process
        import viziphant

        np.random.seed(30)
        spiketrains = compound_poisson_process(rate=15 * pq.Hz,
            amplitude_distribution=[0, 0.95, 0, 0, 0, 0, 0.05], t_stop=5*pq.s)
        bst = BinnedSpikeTrain(spiketrains, bin_size=10 * pq.ms)
        bst.rescale('ms')
        patterns = cell_assembly_detection(bst, max_lag=2)

        viziphant.patterns.plot_patterns(spiketrains, patterns=patterns[:2],
                                         circle_sizes=(3, 30, 40))
        plt.show()

    Additionally, one can add events to the returned axes:

    .. code-block:: python

        event = neo.Event([0.5, 3.8] * pq.s, labels=['Trig ON', 'Trig OFF'])
        viziphant.events.add_event(axes, event=event)

    """
    if isinstance(patterns, dict):
        patterns = [patterns]
    axes = rasterplot(spiketrains, color='darkgray', s=circle_sizes[0])
    units = spiketrains[0].units
    time_scalar = units.rescale('ms').item()
    patterns_overlap = defaultdict(lambda: defaultdict(list))

    if colors is None:
        # +1 is necessary
        cmap = plt.cm.get_cmap("hsv", len(patterns) + 1)
        colors = np.array([cmap(i) for i in range(len(patterns))])
    elif not isinstance(colors, (list, tuple, np.ndarray)):
        raise TypeError("'colors' must be a list of colors")
    elif len(colors) != len(patterns):
        raise ValueError("The length of 'colors' must match the length of "
                         "the input 'patterns'.")

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
    axes.yaxis.set_label_coords(-0.01, 0.5)
    return axes
