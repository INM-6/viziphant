"""
Spike train synchrony plots
---------------------------

.. autosummary::
    :toctree: toctree/spike_train_synchrony

    plot_spike_contrast

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.

import matplotlib.pyplot as plt
import numpy as np

from viziphant.rasterplot import rasterplot


def plot_spike_contrast(trace, spiketrains=None, title=None, lw=1.0,
                        xscale='log', **kwargs):
    """
    Plot Spike-contrast synchrony measure :cite:`Ciba18_136`.

    Parameters
    ----------
    trace : SpikeContrastTrace
        The trace output from
        :func:`elephant.spike_train_synchrony.spike_contrast` function.
    spiketrains : list of neo.SpikeTrain or None
        Input spike trains, optional. If provided, the raster plot will be
        shown at the bottom.
        Default: None
    title : str or None.
        The plot title. If None, an automatic description will be set.
        Default: None
    lw : float, optional
        The curves line width.
        Default: 1.0
    xscale : str, optional
        X axis scale.
        Default: 'log'
    **kwargs
        Additional arguments, passed in :func:`viziphant.rasterplot.rasterplot`

    Returns
    -------
    axes : matplotlib.Axes.axes

    Examples
    --------
    Spike-contrast synchrony of homogenous Poisson processes.

    .. plot::
        :include-source:

        import numpy as np
        import quantities as pq
        from elephant.spike_train_generation import homogeneous_poisson_process
        from elephant.spike_train_synchrony import spike_contrast
        import viziphant
        np.random.seed(24)
        spiketrains = [homogeneous_poisson_process(rate=20 * pq.Hz,
                       t_stop=10 * pq.s) for _ in range(10)]
        synchrony, trace = spike_contrast(spiketrains, return_trace=True)
        viziphant.spike_train_synchrony.plot_spike_contrast(trace,
             spiketrains=spiketrains, c='gray', s=1)
        plt.show()

    """
    nrows = 2 if spiketrains is not None else 1
    fig, axes = plt.subplots(nrows=nrows)
    axes = np.atleast_1d(axes)
    units = trace.bin_size.units
    bin_sizes = trace.bin_size.magnitude
    axes[0].plot(bin_sizes, trace.contrast, lw=lw, label=r'Contrast($\Delta$)',
                 linestyle='dashed', color='limegreen')
    axes[0].plot(bin_sizes, trace.active_spiketrains, lw=lw,
                 label=r'ActiveST($\Delta$)',
                 linestyle='dashdot', color='dodgerblue')
    axes[0].plot(bin_sizes, trace.synchrony, lw=lw,
                 label=r'Synchrony($\Delta$)', color='black')
    bin_id_max = np.argmax(trace.synchrony)
    synchrony_loc = bin_sizes[bin_id_max], trace.synchrony[bin_id_max]
    axes[0].scatter(*synchrony_loc, s=20, c='red', marker='x')
    axes[0].annotate('S', synchrony_loc, color='red', va='bottom', ha='left')
    axes[0].legend()
    axes[0].set_xscale(xscale)
    axes[0].set_xlabel(fr"Bin size $\Delta$ ({units.dimensionality})")
    if title is None:
        title = "Spike-contrast synchrony measure"
    axes[0].set_title(title)
    if spiketrains is not None:
        rasterplot(spiketrains, axes=axes[1], **kwargs)
        axes[1].set_ylabel('neuron')
        axes[1].yaxis.set_label_coords(-0.01, 0.5)
    plt.tight_layout()
    return axes
