"""
Analysis of Sequences of Synchronous EvenTs (ASSET) plots
---------------------------------------------------------
Visualizes the output of :class:`elephant.asset.ASSET` analysis.

.. autosummary::
    :toctree: toctree/asset

    plot_synchronous_events

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.


import numpy as np
import warnings

from viziphant.rasterplot import rasterplot


def plot_synchronous_events(spiketrains, sse, title=None, **kwargs):
    """
    Reorder and plot the `spiketrains` according to a series of synchronous
    events `sse` obtained with the ASSET analysis. Spike trains that do not
    participate in the chosen group will be shown at the top in a different
    color.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        ASSET input spiketrains.
    sse : dict
        One entry of the output dict from
        :meth:`elephant.asset.ASSET.extract_synchronous_events`.
    title : str or None, optional
        User-defined title string. If None, it'll be set to an automatic
        description.
        Default: None
    **kwargs
        Additional arguments to :func:`viziphant.rasterplot.rasterplot`

    Returns
    -------
    axes : matplotlib.Axes.axes

    See Also
    --------
    viziphant.patterns.plot_patterns : plot patterns repeated in time

    Examples
    --------
    In this example we

      * simulate two noisy synfire chains;
      * shuffle the neurons to destroy visual appearance;
      * run ASSET analysis to recover the original neurons arrangement.

    .. plot::
        :include-source:

        import neo
        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt

        import viziphant
        from elephant import asset

        np.random.seed(10)
        spiketrain = np.linspace(0, 50, num=10)
        np.random.shuffle(spiketrain)
        spiketrains = np.c_[spiketrain, spiketrain + 100]
        spiketrains += np.random.random_sample(spiketrains.shape) * 5
        spiketrains = [neo.SpikeTrain(st, units='ms', t_stop=1 * pq.s)
                       for st in spiketrains]
        asset_obj = asset.ASSET(spiketrains, bin_size=3 * pq.ms)

        imat = asset_obj.intersection_matrix()
        pmat = asset_obj.probability_matrix_analytical(imat,
                                                       kernel_width=50 * pq.ms)
        jmat = asset_obj.joint_probability_matrix(pmat, filter_shape=(5, 1),
                                                  n_largest=3)
        mmat = asset_obj.mask_matrices([pmat, jmat], thresholds=.9)
        cmat = asset_obj.cluster_matrix_entries(mmat, max_distance=11,
                                                min_neighbors=3, stretch=5)
        sses = asset_obj.extract_synchronous_events(cmat)

        viziphant.asset.plot_synchronous_events(spiketrains, sse=sses[1], s=10)
        plt.show()

    Refer to `ASSET tutorial
    <https://elephant.readthedocs.io/en/latest/tutorials/asset.html>`_
    for real-case scenario.

    """
    if len(sse) == 0:
        warnings.warn("Passed an empty synchronous event dict.")
    cluster_chain = []
    for chain in sse.values():
        cluster_chain.extend(chain)

    _, indices_pattern = np.unique(cluster_chain, return_index=True)
    indices_pattern = np.take(cluster_chain, np.sort(indices_pattern))
    indices_left = set(range(len(spiketrains))).difference(indices_pattern)

    reordered_sts = [spiketrains[idx] for idx in indices_pattern]
    sts_not_a_pattern = [spiketrains[idx] for idx in sorted(indices_left)]
    if title is None:
        title = "Neurons ordering reconstructed with ASSET"
    axes = rasterplot([reordered_sts, sts_not_a_pattern],
                      title=title, **kwargs)
    axes.set_ylabel('reordered neurons')
    axes.yaxis.set_label_coords(-0.01, 0.5)

    return axes
