'''
This is a local version of the UP task to calculate all pairwise CCHs and their
significance based on 1000 spike dither surrogates on the PBS based cluster.
'''

# =============================================================================
# Initialization
# =============================================================================
import os
import numpy as np
import quantities as pq
# provides neo framework and I/Os to load data
import neo

# provides core analysis library component
import elephant
import elephant.spike_train_correlation as stc
import elephant.conversion as conv

# import other utilities
import time
import h5py_wrapper.wrapper

try:
    job_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
    job_parameter = 1

# =============================================================================
# Load Spinnaker data
# =============================================================================
if job_parameter == 0:
    filename = '../../data/Spinnaker_Data/results/spikes_L5E.h5'
    session = neo.NeoHdf5IO(filename=filename)
    block = session.read_block()
    sts = block.list_children_by_class(neo.SpikeTrain)[:100]
    print("Number of spinnaker spike trains: " + str(len(sts)))


# =============================================================================
# Load Nest data
# =============================================================================
if job_parameter == 1:
    filename = '../../data/Nest_Data/example_output_10500ms_nrec_100/spikes_L5E.h5'
    session = neo.NeoHdf5IO(filename=filename)

    sts = []

    for k in range(100):
        sts.append(session.get("/" + "SpikeTrain_" + str(k)))
    print("Number of nest spike trains: " + str(len(sts)))


# ## Analysis parameters
num_surrs = 1000
lag_res = 1 * pq.ms

cc = {}

# Rates
cc['rate'] = [
    elephant.statistics.mean_firing_rate(st).rescale("Hz").magnitude
    for st in sts]

# CV and LV
isis = [elephant.statistics.isi(st) for st in sts]

cc['cv'] = [elephant.statistics.cv(isi) for isi in isis if len(isi) > 1]
cc['lv'] = [elephant.statistics.lv(isi) for isi in isis if len(isi) > 1]


# original corrcoeff
t0 = time.time()
cco = stc.corrcoef(conv.BinnedSpikeTrain(sts, lag_res))
cc['original_measure'] = cco

print 'Computed corrcoeff'
t1 = time.time()
surr = [elephant.spike_train_surrogates.dither_spike_train(
        st, shift=50. * pq.ms, n=num_surrs) for st in sts]

print 'Generated surrogate'
t2 = time.time()
cco_surr = []
for idx_surr in range(num_surrs):
    cco_surr.append(stc.corrcoef(conv.BinnedSpikeTrain([
        s[idx_surr] for s in surr], lag_res)))

print 'Computed corrcoeff surrogates'
t3 = time.time()
pvalues = np.sum(
    np.array([(cco_s) >= cco for cco_s in cco_surr]), axis=0) / float(num_surrs)
t4 = time.time()
cc['surr'] = np.array(cco_surr)
cc['pvalue'] = pvalues
print 'Corrcoeff data', t1-t0
print 'Gen surrogate', t2-t1
print 'Corrcoeff surr', t3-t2
print 'Pvalues', t4-t3

# write parameters to disk
if job_parameter == 0:
    data = 'spinnaker'
if job_parameter == 1:
    data = 'nest'
filename = '../../results/release_demo/viz_corrcoeff_' + data + '.h5'
if os.path.exists(filename):
    os.remove(filename)
h5py_wrapper.wrapper.add_to_h5(
    filename, cc, write_mode='w', overwrite_dataset=True)
