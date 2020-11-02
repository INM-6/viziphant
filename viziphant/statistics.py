"""
Spike train statistics plots
----------------------------

.. autosummary::
    :toctree: toctree/statistics/

    plot_isi_histogram
    plot_time_histogram

"""
# Copyright 2017-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from elephant import statistics


def plot_isi_histogram(intervals, axes=None, label='', bin_size=3 * pq.ms,
                       cutoff=None):
    """
    Create a simple histogram plot to visualise an inter-spike interval (ISI)
    distribution computed with :func:`elephant.statistics.isi`.

    Parameters
    ----------
    intervals : neo.SpikeTrain or pq.Quantity
        A spiketrain the ISI to be computed from or the intervals themselves
        returned by :func:`elephant.statistics.isi`.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If set to None, new axes are created and
        returned.
        Default: None
    label : str, optional
        The label of the ISI distribution.
        Default: ''
    bin_size : pq.Quantity, optional
        The bin size for the histogram.
        Default: 2 ms
    cutoff : pq.Quantity or None, optional
        The largest ISI to consider. Otherwise, if set to None, all range of
        values are plotted. The typical cutoff value is ~250 ms.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    Examples
    --------
    1. Basic ISI histogram plot.

    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.statistics import plot_isi_histogram
        np.random.seed(12)

        spiketrain = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10*pq.s)
        plot_isi_histogram(spiketrain, cutoff=250*pq.ms)
        plt.show()

    2. Multiple ISI histograms are shown side by side.

    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.statistics import plot_isi_histogram
        np.random.seed(12)
        spiketrain1 = homogeneous_poisson_process(rate=30*pq.Hz,t_stop=50*pq.s)
        spiketrain2 = homogeneous_poisson_process(rate=10*pq.Hz,t_stop=50*pq.s)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        for i, spiketrain in enumerate([spiketrain1, spiketrain2]):
            plot_isi_histogram(spiketrain, axes=axes[i],
                               label=f"Neuron '{i}'", cutoff=250*pq.ms)
        plt.show()

    """
    if isinstance(intervals, neo.SpikeTrain):
        intervals = statistics.isi(spiketrain=intervals)

    if axes is None:
        fig, axes = plt.subplots()

    if cutoff is None:
        cutoff = intervals.max()

    bins = np.arange(0, cutoff.rescale(intervals.units).item(),
                     bin_size.rescale(intervals.units).item())

    axes.hist(intervals, bins=bins)
    axes.set_title(f'{label} ISI distribution')
    axes.set_xlabel(f'Inter-spike interval '
                    f'({intervals.dimensionality.string})')
    axes.set_ylabel('Count')

    return axes


def plot_time_histogram(histogram, axes=None, units=None):
    """
    This function plots a time histogram, such as the result of
    :func:`elephant.statistics.time_histogram`.

    Parameters
    ----------
    histogram : neo.AnalogSignal
        Object containing the histogram bins.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If set to None, new axes are created and
        returned.
    units : pq.Quantity or str or None, optional
        Desired time axis units.
        If None, ``histogram.sampling_period`` units are used.
        Default: None

    Returns
    -------
    axes : matplotlib.axes.Axes

    Examples
    --------
    1. Basic example of spike count histogram.

    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant import statistics
        from viziphant.statistics import plot_time_histogram
        np.random.seed(14)

        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        histogram = statistics.time_histogram(spiketrains, bin_size=100*pq.ms)

        plot_time_histogram(histogram, units='s')
        plt.show()

    2. Multiple time histograms are shown side by side with a common event
       point.

    .. plot::
        :include-source:

        import neo
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant import statistics
        from viziphant.statistics import plot_time_histogram
        from viziphant.events import add_event
        np.random.seed(11)

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        event = neo.Event([2]*pq.s, labels=['Trigger ON'])
        for axis in axes:
            spiketrains = [homogeneous_poisson_process(rate=10 * pq.Hz,
                           t_stop=10 * pq.s) for _ in range(10)]
            histogram = statistics.time_histogram(spiketrains,
                                                  bin_size=0.1 * pq.s,
                                                  output='rate')
            plot_time_histogram(histogram, axes=axis, units='s')
        add_event(axes, event=event)
        plt.show()

    """
    if axes is None:
        fig, axes = plt.subplots()

    # Rescale the time axis if requested
    if units is None:
        units = histogram.sampling_period.units
    elif isinstance(units, str):
        units = pq.Quantity(1, units)
    width = histogram.sampling_period.rescale(units).item()
    times = histogram.times.rescale(units).magnitude

    # Create the plot
    axes.bar(times, histogram.squeeze().magnitude, align='edge', width=width)
    axes.set_xlabel(f"Time ({units.dimensionality})")

    # Human-readable description of the 'output' flag used in time_histogram
    output_dict = dict(counts="Counts",
                       mean="Counts per spike train",
                       rate=f"Spike rate ({histogram.units.dimensionality})")
    normalization = histogram.annotations.get('normalization')
    axes.set_ylabel(output_dict.get(normalization))

    return axes
