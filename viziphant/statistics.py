"""
Simple plotting functions for statistical measures of spike trains
"""

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq

from elephant import statistics


def plot_isi(intervals, label='', bin_size=2 * pq.ms, cutoff=250 * pq.ms):
    """
    This function creates a simple histogram plot to visualise an inter-spike
    interval (ISI) distribution computed with `elephant.statistics.isi`.

    Parameters
    ----------
    intervals : neo.SpikeTrain or pq.Quantity
        A spiketrain the ISI to be computed from or the direct output of
        :func:`elephant.statistics.isi`.
    label : str, optional
        The label of the ISI distribution. Default: ''
    bin_size : pq.Quantity, optional
        The bin size for the histogram. Default: 2 ms
    cutoff : pq.Quantity, optional
        The largest ISI to consider. Default: 250 ms

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
        from viziphant.statistics import plot_isi
        np.random.seed(12)

        spiketrain = homogeneous_poisson_process(rate=10*pq.Hz, t_stop=10*pq.s)
        plot_isi(spiketrain)
        plt.show()

    """
    if isinstance(intervals, neo.SpikeTrain):
        intervals = statistics.isi(spiketrain=intervals)

    fig, ax = plt.subplots(figsize=(8, 3))

    bins = np.arange(0, cutoff.rescale(intervals.units).magnitude.item(),
                     bin_size.rescale(intervals.units).magnitude.item())

    ax.hist(intervals, bins=bins)
    ax.set_title(f'{label} ISI distribution')
    ax.set_xlabel(f'Inter-spike interval ({intervals.dimensionality.string})')
    ax.set_ylabel('Count')

    return fig, ax


def plot_time_histogram(histogram, time_unit=None, y_label=None, max_y=None,
                        event_time=None, event_label=None, **kwargs):
    """
    This function plots a time histogram, such as the result of
    :func:`elephant.statistics.time_histogram`.

    Parameters
    ----------
    histogram : neo.AnalogSignal
        Object containing the histogram bins.
    time_unit : pq.Quantity, optional
        Desired unit for the plot time axis.
        If None, the current unit of `histogram` is used.
        Default: None
    y_label : str, optional
        Label for the Y axis.
        Default: None
    max_y : int or float, optional
        Maximum value for the Y axis.
        Default: None
    event_time : pq.Quantity, optional
        To draw a vertical line showing an event in the plot. The `event_time`
        is provided with respect to the start of the histogram.
        The histogram times will be centered at this time (i.e., the point at
        `event_time` will be zero).
        Default: None
    event_label : str, optional
        Label of the event.
        If None, the label is not plotted.
        If `event_time` is None, this parameter is ignored.
        Default: None

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
        from elephant import statistics
        from viziphant.statistics import plot_time_histogram
        np.random.seed(13)

        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        histogram = statistics.time_histogram(spiketrains, bin_size=100*pq.ms)

        plot_time_histogram(histogram, y_label='counts')
        plt.show()

    """
    fig, ax = plt.subplots(**kwargs)

    # Rescale the time axis if requested
    if time_unit is None:
        width = histogram.sampling_period.rescale(
            histogram.times.units).magnitude
        times = histogram.times.magnitude
        time_unit = histogram.times.units.dimensionality
    else:
        width = histogram.sampling_period.rescale(time_unit).magnitude
        times = histogram.times.rescale(time_unit).magnitude
        time_unit = time_unit.units.dimensionality

    # Shift times according to the event, if provided
    if event_time is not None:
        times = np.subtract(times, event_time.rescale(time_unit).magnitude)

    # Create the plot
    ax.bar(times, histogram.squeeze().magnitude, align='edge', width=width)
    ax.set_xlabel(f"Time ({time_unit})")

    # Define Y label and Y-axis limits
    if max_y is not None:
        ax.set_ylim([0, max_y])

    if y_label is not None:
        ax.set_ylabel(y_label)

    # Add the event line and label, if provided
    if event_time is not None:
        ax.axvline(0, linewidth=1, linestyle='solid', color='black')
        if event_label is not None:
            ax.text(0, ax.get_ylim()[1], event_label,
                    horizontalalignment='center',
                    verticalalignment='bottom')

    return fig, ax
