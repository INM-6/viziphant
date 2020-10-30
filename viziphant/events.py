import neo
import numpy as np
import quantities as pq


def add_event(axes, event, key=None):
    """
    Add event(s) to axes plot.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Axes plot instance.
    event : neo.Event
        A `neo.Event` instance that contains labels or `array_annotations` and
        time points when the event(s) is occurred.
    key : str or None, optional
        If set to not None, the event labels will be extracted from
        ``event.array_annotations[key]``. Otherwise, event labels are extracted
        from ``event.labels``.
        Default: None

    Examples
    --------
    .. plot::
        :include-source:

        import neo
        import quantities as pq
        import matplotlib.pyplot as plt
        from viziphant.events import add_event
        fig, axes = plt.subplots()
        event = neo.Event([1, 6, 9] * pq.s, labels=['trig0', 'trig1', 'trig2'])
        add_event(axes, event=event)
        plt.show()

    Refer to :func:`viziphant.rasterplot.eventplot` for real scenarios.

    """
    axes = np.atleast_1d(axes)
    for event_idx in range(len(event)):
        time = event.times[event_idx].rescale(pq.s).item()
        if key is not None:
            label = event.array_annotations[key][event_idx]
        else:
            label = event.labels[event_idx]
        for axis in axes:
            axis.axvline(time, color='black')
            axis.axvline(time, color='black')
        axes[0].text(time, axes[0].get_ylim()[1], label,
                     horizontalalignment='left',
                     verticalalignment='bottom', rotation=40)
