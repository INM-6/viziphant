"""
Unit tests for the ASSET analysis plot functions.
:copyright: Copyright 2014-2023 by the Viziphant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo
import numpy as np
import quantities as pq

import viziphant
from elephant import asset
class AssetTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(10)
        spiketrain = np.linspace(0, 50, num=10)
        np.random.shuffle(spiketrain)
        spiketrains = np.c_[spiketrain, spiketrain + 100]
        spiketrains += np.random.random_sample(spiketrains.shape) * 5
        self.spiketrains = [neo.SpikeTrain(st, units='ms', t_stop=1 * pq.s)
                       for st in spiketrains]

    def test_plot_synchronous_events(self):
        # Test that the plotting function does not throw an exception
        asset_obj = asset.ASSET(self.spiketrains, bin_size=3 * pq.ms)

        imat = asset_obj.intersection_matrix()
        pmat = asset_obj.probability_matrix_analytical(imat,
                                                       kernel_width=50 * pq.ms)
        jmat = asset_obj.joint_probability_matrix(pmat, filter_shape=(5, 1),
                                                  n_largest=3)
        mmat = asset_obj.mask_matrices([pmat, jmat], thresholds=.9)
        cmat = asset_obj.cluster_matrix_entries(mmat, max_distance=11,
                                                min_neighbors=3, stretch=5)
        sses = asset_obj.extract_synchronous_events(cmat)

        viziphant.asset.plot_synchronous_events(self.spiketrains, sse=sses[1], s=10)
