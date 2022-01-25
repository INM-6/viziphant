"""
Adding time events to axes plot
-------------------------------

.. autosummary::
    :toctree: toctree/events/

    add_event

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.


import neo
import numpy as np
import quantities as pq


def add_event(axes, event, key=None, rotation=40):
    """
    Add event(s) to axes plot.

    If `axes` is a list of Axes, they are assumed to be top-down aligned, and
    the annotation text will be displayed on the first (uppermost) axis.

    Original input ``event.times`` units are used. If you want to use units
    other than the inputs, e.g. milliseconds, rescale the event manually by
    performing ``event = event.rescale('ms')``.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or list
        Matplotlib Axes handle or list of Axes.
    event : neo.Event
        A `neo.Event` instance that contains labels or `array_annotations` and
        time points when the event(s) is occurred.
    key : str or None, optional
        If set to not None, the event labels will be extracted from
        ``event.array_annotations[key]``. Otherwise, event labels are extracted
        from ``event.labels``.
        Default: None
    rotation : int, optional
        Text label rotation in degrees.
        Default : 40

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
    times = event.times.magnitude
    for event_idx, time in enumerate(times):
        if key is None:
            label = event.labels[event_idx]
        else:
            label = event.array_annotations[key][event_idx]
        for axis in axes:
            axis.axvline(time, color='black')
        axes[0].text(time, axes[0].get_ylim()[1], label,
                     horizontalalignment='left',
                     verticalalignment='bottom', rotation=rotation)
