# Copyright 2017-2020 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt.txt for details.

import quantities as pq


def check_same_units(quantities):
    if not isinstance(quantities, (list, tuple)):
        # a single neo object
        return
    for quantity in quantities:
        if not isinstance(quantity, pq.Quantity):
            raise TypeError(f"The input must be a list of neo objects or"
                            f"quantities. Got {type(quantity)}")
        if quantity.units != quantities[0].units:
            raise ValueError("The input quantities must have the same units, "
                             "which is achieved with object.rescale('ms') "
                             "operation.")
