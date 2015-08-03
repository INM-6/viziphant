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
t_stop = 10*pq.s
N = 100
M = 20
rate = 10 * pq.Hz
rate_c = 4 * pq.Hz
pi = 0.8
ro = 0.1

# =============================================================================
# Generate and store independent data
# =============================================================================


sts_ind = stoc.poisson(rate, t_stop=t_stop, n=N)
for st in sts_ind:
    st.annotate(use_st=True)
block_ind = neo.Block()
sgm_ind = neo.Segment()
for st in sts_ind:
    sgm_ind.spiketrains.append(st)
    st.segment = sgm_ind
block_ind.segments.append(sgm_ind)


# save independent
filename = '../../data/independent_2.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_ind)
session.close()

# =============================================================================
# Generate and store SIP data
# =============================================================================

sts_sip = stoc.sip_poisson(
    M=M, N=N-M, rate_b=rate, rate_c=rate_c, T=t_stop)
for st in sts_sip:
    st.annotate(use_st=True)
block_sip = neo.Block()
sgm_sip = neo.Segment()
for st in sts_sip:
    sgm_sip.spiketrains.append(st)
    st.segment = sgm_sip
block_sip.segments.append(sgm_sip)


# save sip
filename = '../../data/sip_3.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_sip)
session.close()

# =============================================================================
# Generate and store MIP data
# =============================================================================

sts_mip = stoc.mip_gen(
    M=M, N=N-M, rate_b=rate, pi=pi, rate_c=rate_c, t_stop=t_stop)
for st in sts_mip:
    st.annotate(use_st=True)
block_mip = neo.Block()
sgm_mip = neo.Segment()
for st in sts_mip:
    sgm_mip.spiketrains.append(st)
    st.segment = sgm_mip
block_mip.segments.append(sgm_mip)


# save mip
filename = '../../data/mip.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_mip)
session.close()

# =============================================================================
# Generate and store CPP data
# =============================================================================

sts_cpp = stoc.cpp_corrcoeff(ro=ro, xi=M, N=N, rate=rate, t_stop=t_stop)
for st in sts_cpp:
    st.annotate(use_st=True)
block_cpp = neo.Block()
sgm_cpp = neo.Segment()
for st in sts_cpp:
    sgm_cpp.spiketrains.append(st)
    st.segment = sgm_cpp
block_cpp.segments.append(sgm_cpp)


# save cpp
filename = '../../data/cpp.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_cpp)
session.close()
