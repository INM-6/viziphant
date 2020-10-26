"""
Simple plotting functions for statistical measures of spike trains
"""

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq


def plot_isi(intervals, label, binsize=2*pq.ms, cutoff=250*pq.ms):
    """
    This function creates a simple histogram plot to visualise an inter-spike
    interval (ISI) distribution computed with `elephant.statistics.isi`.

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


def plot_time_histogram(histogram, time_unit=None, y_label=None, max_y=None,
                        event_time=None, event_label=None, **kwargs):
    """
    This function plots a time histogram, such as the result of
    `elephant.statistics.time_histogram`.

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


def plot_instantaneous_rates(rates, sampling_period, t_start, t_stop,
                             events=None, event_labels=None):
    """
    Plots a list of instantaneous firing rates. Each item is the rate of a
    single spike train.

    This function helps visualizing the result of
    `elephant.statistics.instantaneous_rate` for several neurons.

    Parameters
    ----------
    rates : list of neo.AnalogSignal
        List with the rates calculated for each neuron (using
        `elephant.statistics.instantaneous_rate`).
    sampling_period : pq.Quantity
        Same parameter passed to `elephant.statistics.instantaneous_rate`.
    t_start : pq.Quantity
        Start time of the spike trains used to calculate the rates.
    t_stop : pq.Quantity
        Stop time of the spike trains used to calculate the rates.
    events : neo.Event, optional
        If provided, the events will be added to the plot.
        Default: None
    event_labels : str, optional
        Array annotations from which the trial labels are obtained.
        If None, no labels are used.
        If `events` is None, this parameter is ignored.
        Default: None

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # Convert list to array
    all_rates_array = np.asarray(rates)

    # If list with more than one element, the array shape will be
    # (n_neurons, n_samples, 1)
    if all_rates_array.ndim == 3 and all_rates_array.shape[2] == 1:
        # Strip out the last dimension
        all_rates_array = all_rates_array[:, :, 0]
    else:
        # Transpose to get correct orientation, as the shape will be
        # (n_samples, 1)
        all_rates_array = all_rates_array.T

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.pcolormesh(all_rates_array, cmap='rainbow', shading='flat')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Firing rate [Hz]")

    # Compute time axis ticks
    start = t_start.rescale(pq.s).magnitude
    stop = t_stop.rescale(pq.s).magnitude
    step = sampling_period.rescale(pq.s).magnitude

    times = np.arange(start, stop,
                      (500 * pq.ms / sampling_period).magnitude * step)
    ticks = np.linspace(0, all_rates_array.shape[1], len(times))

    # Define X label and ticks
    ax.set_xlabel("Time (s)")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{l:.1f}" for l in times]);

    # Define Y label and ticks (offset by 0.5 to center the lines on the
    # integer)
    n_neurons = all_rates_array.shape[0]
    order_of_magnitude = np.floor(np.log10(n_neurons))
    step_y = 1 if n_neurons < 10 else int(5**order_of_magnitude)

    ax.set_ylabel("Neuron")
    ax.set_yticks(np.arange(0.5, n_neurons + 0.5, step_y))
    ax.set_yticklabels(np.arange(0, n_neurons, step_y))

    if events is not None:
        unit = pq.CompoundUnit(
            "{}*s".format(sampling_period.rescale('s').item()))

        # Add vertical lines for events
        for event_idx in range(len(events)):
            time = events.times[event_idx].rescale(unit)
            ax.axvline(time, color='black', linewidth=1)
            if event_labels is not None:
                label = events.array_annotations[event_labels][event_idx]
                ax.text(time, ax.get_ylim()[1], label,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=40)

    return fig, ax
