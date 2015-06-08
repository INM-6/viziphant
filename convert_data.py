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
import rg.restingstateio
import mesocircuitio


# =============================================================================
# Global variables
# =============================================================================

# duration of recording to load
rec_start = 10.*pq.s
duration = 50.*pq.s


# =============================================================================
# Load experimental data
# =============================================================================

# data should be in a subdirectory 'data' relative to this notebook's location
# Load only first unit (ID: 1) of each channel

session_exp = rg.restingstateio.RestingStateIO(
    "data/i140701-004", print_diagnostic=False)
block_exp = session_exp.read_block(
    n_starts=[rec_start], n_stops=[rec_start + duration],
    channel_list=[], units=[1])

# select spike trains (min. 2 spikes, SUA only)
sts_exp = [
    st for st in
    block_exp.filter(sua=True, object="SpikeTrain") if len(st) > 2]

for st in sts_exp:
    st.annotate(use_st=True)

print("Number of experimental spike trains: " + str(len(sts_exp)))

# =============================================================================
# Load simulation data
# =============================================================================

# data should be in a subdirectory 'data' relative to this notebook's location
# Load only first unit (ID: 0) of each channel (one exc., one inh.) in layer 5
session_mdl = mesocircuitio.MesoCircuitIO(
    "data/utah_array_spikes_60s.h5", print_diagnostic=False)
block_mdl = session_mdl.read_block(
    n_starts=[10 * pq.s], n_stops=[10 * pq.s + duration],
    channel_list=[], layer_list=['L5'],
    units=[], unit_type=['excitatory', 'inhibitory'])

# Select one neuron for each channel containing a neuron in experimental data
sts_mdl = []
for st in sts_exp:
    sts_mdl.append(block_mdl.filter(
        [{'channel_id': st.annotations['ca_id']}])[0])

# Alternative: randomly pick the same number of excitatory units
# sts_mdl = block_mdl.filter(
#     targdict=[{'unit_type': 'excitatory'}, {'unit_id': 0}])
# sts_mdl = [
#     sts_mdl[i] for i in np.linspace(
#         0, len(sts_mdl) - 1, len(sts_exp), dtype=int)]


for st in sts_mdl:
    st.annotate(use_st=True)
    st.annotate(ca_id=st.annotations['channel_id'])
    st.unit.annotate(ca_id=st.annotations['channel_id'])

print("Number of model spike trains: " + str(len(sts_mdl)))

# save experimental
filename = 'data/experiment.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_exp)
session.close()

# save model
filename = 'data/model.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_mdl)
session.close()
