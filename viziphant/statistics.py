"""
Spike train statistics plots
----------------------------

.. autosummary::
    :toctree: toctree/statistics/

    plot_isi_histogram
    plot_time_histogram
    plot_instantaneous_rates_colormesh

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from elephant import statistics
from viziphant.utils import check_same_units


def plot_isi_histogram(spiketrains, axes=None, bin_size=3 * pq.ms, cutoff=None,
                       title='ISI distribution', legend=None, histtype='step'):
    """
    Create a simple histogram plot to visualise an inter-spike interval (ISI)
    distribution of spike trains.

    Input spike trains are sorted in time prior to computing the ISI.

    If the input is a list of list of spike trains, as in the Example 3, the
    ISI of a population is concatenated from all neuron spike trains.

    Parameters
    ----------
    spiketrains : neo.SpikeTrain or pq.Quantity or list
        A spike train or a list of spike trains the ISI to be computed from.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If set to None, new axes are created and
        returned.
        Default: None
    bin_size : pq.Quantity, optional
        The bin size for the histogram.
        Default: 3 ms
    cutoff : pq.Quantity or None, optional
        The largest ISI to consider. Otherwise, if set to None, all range of
        values are plotted. Typical cutoff values are ~250 ms.
        Default: None
    title : str, optional
        The axes title.
        Default: 'ISI distribution'
    legend : str or list of str or None, optional
        The axes legend labels.
        Default: None
    histtype : str
        Histogram type passed to matplotlib `hist` function.
        Default: 'step'

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

        spiketrain = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=50*pq.s)
        plot_isi_histogram(spiketrain, cutoff=250*pq.ms, histtype='bar')
        plt.show()

    2. ISI histogram of multiple spike trains.

    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.statistics import plot_isi_histogram
        np.random.seed(12)
        rates = [5, 10, 15] * pq.Hz
        spiketrains = [homogeneous_poisson_process(rate=r,
                       t_stop=100 * pq.s) for r in rates]
        plot_isi_histogram(spiketrains, cutoff=250*pq.ms,
                           legend=rates)
        plt.show()

    3. ISI histogram of multiple neuron populations.

    .. plot::
        :include-source:

        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.statistics import plot_isi_histogram

        np.random.seed(12)
        population1 = [homogeneous_poisson_process(rate=30 * pq.Hz,
                       t_stop=50 * pq.s) for _ in range(10)]
        population2 = [homogeneous_poisson_process(rate=r * pq.Hz,
                       t_stop=50 * pq.s) for r in range(1, 20)]
        plot_isi_histogram([population1, population2], cutoff=250 * pq.ms,
                           legend=['population1', 'population2'])
        plt.show()

    """
    def isi_population(spiketrain_list):
        return [statistics.isi(np.sort(st.magnitude))
                for st in spiketrain_list]

    if isinstance(spiketrains, pq.Quantity):
        spiketrains = [spiketrains]
    check_same_units(spiketrains)

    if isinstance(spiketrains[0], (list, tuple)):
        intervals = [np.hstack(isi_population(sts)) for sts in spiketrains]
        units = spiketrains[0][0].units
    else:
        intervals = isi_population(spiketrains)
        units = spiketrains[0].units

    if legend is None:
        legend = [None] * len(intervals)
    elif isinstance(legend, str):
        legend = [legend]
    if len(legend) != len(intervals):
        raise ValueError("The length of the input list and legend labels do "
                         "not match.")

    if cutoff is None:
        cutoff = max(interval.max() for interval in intervals) * units

    if axes is None:
        fig, axes = plt.subplots()

    bins = np.arange(start=0,
                     stop=(cutoff + bin_size).rescale(units).item(),
                     step=bin_size.rescale(units).item())

    for label, interval in zip(legend, intervals):
        axes.hist(interval, bins=bins, histtype=histtype, label=label)
    axes.set_title(title)
    axes.set_xlabel(f'Inter-spike interval ({units.dimensionality})')
    axes.set_ylabel('Count')
    if legend[0] is not None:
        axes.legend()

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


def plot_instantaneous_rates_colormesh(rates, axes=None, units=None, **kwargs):
    """
    Plots a colormesh of instantaneous firing rates. Each row represents a
    spike train the instantaneous rate was computed from.

    Parameters
    ----------
    rates : neo.AnalogSignal
        `neo.AnalogSignal` matrix of shape ``(len(spiketrains), time)``
        containing instantaneous rates obtained by
        :func:`elephant.statistics.instantaneous_rate` function.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If set to None, new axes are created and
        returned.
    units : pq.Quantity or str or None, optional
        Desired time axis units.
        If None, ``histogram.sampling_period`` units are used.
        Default: None
    **kwargs
        Additional parameters passed to matplotlib `pcolormesh` function.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    .. plot::
        :include-source:

        import quantities as pq
        from elephant import statistics, kernels
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.statistics import plot_instantaneous_rates_colormesh
        np.random.seed(6)
        spiketrains = [homogeneous_poisson_process(rate=10 * pq.Hz,
                       t_stop=10 * pq.s) for _ in range(10)]
        kernel = kernels.GaussianKernel(sigma=100 * pq.ms)
        rates = statistics.instantaneous_rate(spiketrains,
                                              sampling_period=10 * pq.ms,
                                              kernel=kernel)
        plot_instantaneous_rates_colormesh(rates)
        plt.show()

    """
    if axes is None:
        fig, axes = plt.subplots()

    if units is None:
        units = rates.sampling_period.units
    elif isinstance(units, str):
        units = pq.Quantity(1, units)
    t_stop = rates.t_stop.rescale(units).item()
    times = np.r_[rates.times.rescale(units).magnitude, t_stop]
    neurons_range = range(rates.shape[1] + 1)

    im = axes.pcolormesh(times, neurons_range, rates.magnitude.T, **kwargs)

    # Add a colorbar
    cbar = plt.colorbar(im, ax=axes)
    cbar.set_label("Firing rate [Hz]")

    axes.set_xlabel(f"Time ({units.dimensionality})")
    axes.set_ylabel("Neuron")
    axes.set_yticks([rates.shape[1] - 0.5])
    axes.set_yticklabels([rates.shape[1] - 1])

    return axes
