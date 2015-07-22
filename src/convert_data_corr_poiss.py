'''
This script load experimental and simulated data, selects the spike trains to
use in each for analysis, marks these with the annotation "use_st'=True.
Finally, it saves the modified Neo Block as hdf5 for use, e.g., in the UP task.

The underlying idea is that in this process, data and meta data are combined
via the custom IOs (experiment: reachgrasp IO based on Neo's Blackrock IO;
model: mesocircuitio), and the result is saved in a common Neo format.
'''

# =============================================================================
# Initialization
# =============================================================================

import os
import sys

# paths
# to find our "special" elephant
sys.path.insert(1, '..')
# change this to point to your reachgrasp IO
sys.path.insert(1, '../../dataset_repos/reachgrasp/python')
sys.path.insert(1, '../../toolboxes/py/python-neo')
sys.path.insert(1, '../../toolboxes/py/python-odml')
sys.path.insert(1, '../../toolboxes/py/csn_toolbox')
import numpy as np
import quantities as pq

# provides neo framework and I/Os to load exp and mdl data
import neo


# provides the framework to generate simulated data
import jelephant.core.stocmod as stoc


# =============================================================================
# Global variables
# =============================================================================

# duration of recording to load
t_start = 0.*pq.s
duration = 10*pq.s
N = 25
M = 20
rate = 10 * pq.Hz
rate_c = 3 * pq.Hz

# =============================================================================
# Generate and store independent data
# =============================================================================


sts_ind = stoc.poisson(rate, t_stop=duration, n=N)
for st in sts_ind:
    st.annotate(use_st=True)
block_ind = neo.Block()
sgm_ind = neo.Segment()
for st in sts_ind:
    sgm_ind.spiketrains.append(st)
    st.segment=sgm_ind
block_ind.segments.append(sgm_ind)


# =============================================================================
# Generate and store SIP data
# =============================================================================

sts_sip = stoc.sip_poisson(
    M=25, N=N-M, rate_b=rate, rate_c=3*pq.Hz, T=10*pq.s)
for st in sts_sip:
    st.annotate(use_st=True)
block_sip = neo.Block()
sgm_sip = neo.Segment()
for st in sts_sip:
    sgm_sip.spiketrains.append(st)
    st.segment=sgm_sip
block_sip.segments.append(sgm_sip)


# save experimental
filename = '../../data/independent_1.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_ind)
session.close()

# save model
filename = '../../data/sip_1.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_sip)
session.close()
