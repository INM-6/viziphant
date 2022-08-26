"""
Spike train correlation plots
-----------------------------

.. autosummary::
    :toctree: toctree/spike_train_correlation/

    plot_corrcoef
    plot_cross_correlation_histogram

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.


from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from elephant.utils import check_neo_consistency


def plot_corrcoef(corrcoef_matrix, axes=None, correlation_range='auto',
                  colormap='bwr', colorbar_aspect=20,
                  colorbar_padding_fraction=0.5, remove_diagonal=True):
    """
    Plots a cross-correlation matrix returned by
    :func:`elephant.spike_train_correlation.correlation_coefficient`
    function with a color bar.

    Parameters
    ----------
    corrcoef_matrix : np.ndarray
        Pearson's correlation coefficient matrix
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None
    correlation_range : {'auto', 'full'} or tuple of float, optional
        Minimum and maximum correlations to consider for color mapping.
        If tuple, the first element is the minimum and the second
        element is the maximum correlation.
        If 'auto', the maximum absolute value of the non-diagonal coefficients
        will be used symmetrically as minimum and maximum.
        If 'full', maximum correlation is set at 1.0 and minimum at -1.0.
        Default: 'auto'
    colormap : str, optional
        Colormap. Default: 'bwr'
    colorbar_aspect : float, optional
        Aspect ratio of the color bar. Default: 20
    colorbar_padding_fraction : float, optional
        Padding between matrix plot and color bar relative to color bar width.
        Default: 0.5
    remove_diagonal : bool, optional
        If True, the values in the main diagonal are replaced with zeros.
        Default: True

    Returns
    -------
    axes : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If `correlation_range` is not tuple or 'auto' or 'full'.

    Examples
    --------
    Create 10 homogeneous random Poisson spike trains of rate `10Hz` and bin
    the spikes into bins of `100ms` width, which is relatively large for such
    a firing rate, so we expect non-zero correlations.

    .. plot::
       :include-source:

        import quantities as pq
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant.conversion import BinnedSpikeTrain
        from elephant.spike_train_correlation import correlation_coefficient
        from viziphant.spike_train_correlation import plot_corrcoef
        np.random.seed(0)

        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        binned_spiketrains = BinnedSpikeTrain(spiketrains, bin_size=100*pq.ms)
        corrcoef_matrix = correlation_coefficient(binned_spiketrains)

        fig, axes = plt.subplots()
        plot_corrcoef(corrcoef_matrix, axes=axes)
        axes.set_xlabel('Neuron')
        axes.set_ylabel('Neuron')
        axes.set_title("Correlation coefficient matrix")
        plt.show()

    """
    if axes is None:
        fig, axes = plt.subplots()

    if remove_diagonal:
        corrcoef_matrix = corrcoef_matrix.copy()
        np.fill_diagonal(corrcoef_matrix, val=0)

    # Get limits
    if correlation_range == 'full':
        vmin, vmax = -1, 1
    elif correlation_range == 'auto':
        vmax = np.max(np.abs(corrcoef_matrix))
        vmin = -vmax
    elif isinstance(correlation_range, (tuple, list)):
        vmin, vmax = correlation_range
    else:
        raise ValueError(f"Invalid 'correlation_range' ({correlation_range}). "
                         f"Must be a tuple of float values or 'auto'/'full'.")

    image = axes.imshow(corrcoef_matrix, vmin=vmin, vmax=vmax, cmap=colormap)

    # Initialise colour bar axis
    divider = make_axes_locatable(axes)
    width = axes_size.AxesY(axes, aspect=1. / colorbar_aspect)
    pad = axes_size.Fraction(colorbar_padding_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)

    plt.colorbar(image, cax=cax)

    return axes


def plot_cross_correlation_histogram(cch, axes=None, units=None, maxlag=None,
                                     legend=None,
                                     title='Cross-correlation histogram'):
    """
    Plot a cross-correlation histogram returned by
    :func:`elephant.spike_train_correlation.cross_correlation_histogram`,
    rescaled to seconds.

    Parameters
    ----------
    cch : neo.AnalogSignal or list of neo.AnalogSignal
        Cross-correlation histogram or a list of such.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If set to None, new axes are created and
        returned.
        Default: None
    units : pq.Quantity or str or None, optional
        Desired time axis units.
        If None, ``cch.sampling_period`` units are used.
        Default: None
    maxlag : pq.Quantity or None, optional
        Left and right borders of the plot.
        Default: None
    legend : str or list of str or None, optional
        The axes legend labels.
        Default: None
    title : str, optional
        The axes title.
        Default: 'Cross-correlation histogram'

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
        from elephant.conversion import BinnedSpikeTrain
        from elephant.spike_train_correlation import \
             cross_correlation_histogram
        from viziphant.spike_train_correlation import \
             plot_cross_correlation_histogram

        spiketrain1 = homogeneous_poisson_process(rate=10*pq.Hz,
                                                  t_stop=10*pq.s)
        spiketrain2 = homogeneous_poisson_process(rate=10*pq.Hz,
                                                  t_stop=10*pq.s)
        binned_spiketrain1 = BinnedSpikeTrain(spiketrain1, bin_size=100*pq.ms)
        binned_spiketrain2 = BinnedSpikeTrain(spiketrain2, bin_size=100*pq.ms)
        cch, lags = cross_correlation_histogram(binned_spiketrain1,
                                                binned_spiketrain2)

        plot_cross_correlation_histogram(cch)
        plt.show()

    """
    if axes is None:
        fig, axes = plt.subplots()

    if isinstance(cch, neo.AnalogSignal):
        cch = [cch]

    check_neo_consistency(cch, object_type=neo.AnalogSignal)
    if units is None:
        units = cch[0].sampling_period.units
    elif isinstance(units, str):
        units = pq.Quantity(1, units)

    if legend is None:
        legend = [None] * len(cch)
    elif isinstance(legend, str):
        legend = [legend]
    if len(legend) != len(cch):
        raise ValueError("The length of the input list and legend labels do "
                         "not match.")

    for label, signal in zip(legend, cch):
        cch_times = signal.times.rescale(units).magnitude
        axes.plot(cch_times, signal.magnitude, label=label)

    axes.set_ylabel(cch[0].annotations['cch_parameters']['normalization'])
    axes.set_xlabel(f"Time lag ({units.dimensionality})")
    axes.set_title(title)
    if maxlag is not None:
        maxlag = maxlag.rescale(units).magnitude
        axes.set_xlim(-maxlag, maxlag)
    if legend[0] is not None:
        axes.legend()

    return axes
