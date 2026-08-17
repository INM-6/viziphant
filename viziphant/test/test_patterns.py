"""
Unit tests for the pattern plot functions
:copyright: Copyright 2014-2023 by the Viziphant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import neo
import numpy as np
import pytest
import quantities as pq
import viziphant


@pytest.fixture
def spiketrains_and_patterns():
    """
    Build minimal inputs for `plot_patterns`: 5 neo.SpikeTrain objects and
    2 pattern dicts (each with 'neurons' and 'times' keys, as returned by
    elephant's spade/cell_assembly_detection).
    """
    spiketrains = [
        neo.SpikeTrain(
            np.linspace(0, 50, num=10),
            units="ms",
            t_stop=100 * pq.ms,
        )
        for _ in range(5)
    ]
    patterns = [
        {"neurons": [0, 1, 2], "times": [5, 15] * pq.ms},
        {"neurons": [2, 3], "times": [10, 20] * pq.ms},
    ]
    return spiketrains, patterns


def test_plot_patterns_default_colors(spiketrains_and_patterns):
    """
    Regression test for plt.cm.get_cmap -> plt.get_cmap
    """
    spiketrains, patterns = spiketrains_and_patterns
    axes = viziphant.patterns.plot_patterns(spiketrains, patterns)
    assert axes is not None


def test_plot_patterns_colors_length_mismatch(spiketrains_and_patterns):
    """
    Checks if plot_patterns raises value error on color lengh mismatch
    """
    spiketrains, patterns = spiketrains_and_patterns
    with pytest.raises(
        ValueError,
        match=r"The length of \'colors\' must match the length of "
        r"the input \'patterns\'\.",
    ):
        viziphant.patterns.plot_patterns(spiketrains, patterns, colors=["r"])
