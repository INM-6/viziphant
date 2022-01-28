# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.

from elephant.utils import check_same_units as check_same_units_single


def check_same_units(spiketrains):
    if isinstance(spiketrains[0], (list, tuple)):
        for sts in spiketrains:
            # check within a population
            check_same_units_single(sts)
        # check that units match across populations
        check_same_units_single([sts[0] for sts in spiketrains])
    else:
        # a list of neo.SpikeTrain
        check_same_units_single(spiketrains)
