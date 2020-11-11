"""
Viziphant is a package for the visualization of the analysis results from
Elephant, which is a package for the analysis of neurophysiological data,
based on Neo.
"""
# Copyright 2017-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.

from . import (events, gpfa, rasterplot, spade, spike_train_correlation,
               statistics, unitary_event_analysis)


def _get_version():
    import os
    viziphant_dir = os.path.dirname(__file__)
    with open(os.path.join(viziphant_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


__version__ = _get_version()
