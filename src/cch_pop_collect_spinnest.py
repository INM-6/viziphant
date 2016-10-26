'''
This script collects the results of individual runs of hbp_review_task (run
either on UP or locally), and assembles a single, compressed result file (with
some additional info).
'''

# =============================================================================
# Initialization
# =============================================================================

import os
import glob
import pickle

import numpy as np
import quantities as pq

# provides neo framework and I/Os to load ind and nest data
import neo

# provides core analysis library component
import elephant

import h5py_wrapper.wrapper


def cch_measure(cch_all_pairs, times):
    ind = np.argmin(np.abs(times))
    return np.sum(cch_all_pairs[ind - 5:ind + 5])


sts_spinnaker = []
sts_nest = []
# =============================================================================
# Load Spinnaker data
# =============================================================================
#populations = ['L23E','L23I','L4E','L4I','L5E','L5I','L6E','L6I']
#for pop in populations:
#    filename = '../../data/Spinnaker_Data/spikes_{}.h5'.format(pop)
#    session = neo.NeoHdf5IO(filename=filename)
#    block = session.read_block()
#    sts_spinnaker.extend(block.list_children_by_class(neo.SpikeTrain)[:20])
#
#
## =============================================================================
## Load Nest data
## =============================================================================
#
#    filename = '../../data/NEST_Data/spikes_{}.h5'.format(pop)
#    session = neo.NeoHdf5IO(filename=filename)
#    block = session.read_block()
#    sts_nest.extend(block.list_children_by_class(neo.SpikeTrain)[:20])
##    for k in range(100):
##        sts_nest.append(session.get("/" + "SpikeTrain_" + str(k)))
#
#print("Number of spinnaker spike trains: " + str(len(sts_spinnaker)))
#print("Number of nest spike trains: " + str(len(sts_nest)))
## create binned spike trains
#sts_spinnaker_bin = elephant.conversion.BinnedSpikeTrain(
#    sts_spinnaker, binsize=1 * pq.ms)

#num_neurons = len(sts_spinnaker)
num_neurons = 160

## create binned spike trains
#sts_nest_bin = elephant.conversion.BinnedSpikeTrain(
#    sts_nest, binsize=1 * pq.ms)
#
#
## =============================================================================
## Calculate measures
## =============================================================================
#
#rates = {}
#rates['spinnaker'] = [
#    elephant.statistics.mean_firing_rate(st).rescale("Hz").magnitude
#    for st in sts_spinnaker]
#rates['nest'] = [
#    elephant.statistics.mean_firing_rate(st).rescale("Hz").magnitude
#    for st in sts_nest]
#
#isis_spinnaker = [elephant.statistics.isi(st) for st in sts_spinnaker]
#isis_nest = [elephant.statistics.isi(st) for st in sts_nest]
#
#cvs = {}
#cvs['spinnaker'] = [elephant.statistics.cv(isi) for isi in isis_spinnaker if len(isi)>1]
#cvs['nest'] = [elephant.statistics.cv(isi) for isi in isis_nest if len(isi)>1]
#
#lvs = {}
#lvs['spinnaker'] = [elephant.statistics.lv(isi) for isi in isis_spinnaker  if len(isi)>1]
#lvs['nest'] = [elephant.statistics.lv(isi) for isi in isis_nest  if len(isi)>1]
#
#
## =============================================================================
## Rewrite files
## =============================================================================

num_edges = 0
for ni in range(num_neurons):
    for nj in range(ni, num_neurons):
        num_edges += 1

cc = {}
for dta in ['spinnaker', 'nest']:
    cc[dta] = {}

    cc[dta]['meta'] = {}

    cc[dta]['neuron_topo'] = {}
    cc[dta]['neuron_topo']['x'] = np.zeros(num_neurons)
    cc[dta]['neuron_topo']['y'] = np.zeros(num_neurons)

    cc[dta]['func_conn'] = {}
    cc[dta]['func_conn']['cch_peak'] = {}
    cc[dta]['func_conn']['cch_peak']['pvalue'] = np.zeros(num_edges)

    cc[dta]['edges'] = {}
    cc[dta]['edges']['id_i'] = np.zeros(num_edges)
    cc[dta]['edges']['id_j'] = np.zeros(num_edges)

    cc[dta]['neuron_single_values'] = {}
    cc[dta]['neuron_single_values']['rate'] = np.zeros(num_neurons)
    cc[dta]['neuron_single_values']['cv'] = np.zeros(num_neurons)
    cc[dta]['neuron_single_values']['lv'] = np.zeros(num_neurons)
    cc[dta]['neuron_single_values']['behavior'] = np.zeros(num_neurons)

    cc[dta]['edge_time_series'] = {}
    cc[dta]['edge_time_series']['cch_all_pairs'] = None
    cc[dta]['edge_time_series']['sig_upper_975'] = None
    cc[dta]['edge_time_series']['sig_lower_25'] = None
    cc[dta]['edge_time_series']['times_ms'] = None

    cc[dta]['meta']['num_neurons'] = num_neurons
    cc[dta]['meta']['num_edges'] = num_edges
#
## values per neuron
#for dta, sts in zip(['spinnaker', 'nest'], [sts_spinnaker, sts_nest]):
#    for neuron_i in range(num_neurons):
#        lin_channel = neuron_i
#        cc[dta]['neuron_topo']['x'][neuron_i] = \
#            int(lin_channel) / 10
#        cc[dta]['neuron_topo']['y'][neuron_i] = \
#            int(lin_channel) % 10
#
#    cc[dta]['neuron_single_values']['rate'] = rates[dta]
#    cc[dta]['neuron_single_values']['cv'] = cvs[dta]
#    cc[dta]['neuron_single_values']['lv'] = lvs[dta]

# values per edge
num_tasks = len(glob.glob(
    '../../results/release_demo/correlation_output_pop*.h5'))
for job_parameter in range(num_tasks):
    filename = \
        '../../results/release_demo/correlation_output_pop_' + \
        str(job_parameter) + '.h5'
    if not os.path.exists(filename):
        raise IOError('Cannot find file %s.', filename)
    print("Assembly of : %s" % filename)

    cc_part = h5py_wrapper.wrapper.load_h5(filename)

    for dta, sts in zip(['spinnaker', 'nest'], [sts_spinnaker, sts_nest]):
        for calc_i in cc_part[dta]['pvalue']:
            print(
                "Processing %s-%i (%i,%i)" %
                (dta, calc_i,
                    cc_part[dta]['unit_i'][calc_i],
                    cc_part[dta]['unit_j'][calc_i]))
            cc[dta]['func_conn']['cch_peak']['pvalue'][calc_i] = \
                cc_part[dta]['pvalue'][calc_i]

            cc[dta]['edges']['id_i'][calc_i] = cc_part[dta]['unit_i'][calc_i]
            cc[dta]['edges']['id_j'][calc_i] = cc_part[dta]['unit_j'][calc_i]
            if cc[dta]['edge_time_series']['cch_all_pairs'] is None:
                cc[dta]['edge_time_series']['cch_all_pairs'] = np.zeros((
                    num_edges, len(cc_part[dta]['times_ms'][calc_i])))
                cc[dta]['edge_time_series']['sig_upper_975'] = np.zeros((
                    num_edges, len(cc_part[dta]['times_ms'][calc_i])))
                cc[dta]['edge_time_series']['sig_lower_25'] = np.zeros((
                    num_edges, len(cc_part[dta]['times_ms'][calc_i])))
                cc[dta]['edge_time_series']['times_ms'] = np.zeros((
                    num_edges, len(cc_part[dta]['times_ms'][calc_i])))

            # remove if not required anymore!
            ccm = np.zeros(cc_part[dta]['surr'][calc_i].shape[0])
            for xi in range(cc_part[dta]['surr'][calc_i].shape[0]):
                ccm[xi] = cch_measure(
                    cc_part[dta]['surr'][calc_i][xi, :],
                    cc_part[dta]['times_ms'][calc_i])
            smas = np.argsort(ccm)
            cc[dta]['edge_time_series']['cch_all_pairs'][calc_i, :] = \
                cc_part[dta]['original'][calc_i].T
            cc[dta]['edge_time_series']['sig_upper_975'][calc_i, :] = \
                cc_part[dta]['surr'][calc_i][smas[975], :].T
            cc[dta]['edge_time_series']['sig_lower_25'][calc_i, :] = \
                cc_part[dta]['surr'][calc_i][smas[25], :].T
            cc[dta]['edge_time_series']['times_ms'][calc_i, :] = \
                cc_part[dta]['times_ms'][calc_i].T

    del cc_part

# write parameters to disk
filename = '../../results/release_demo/viz_output_pop_spinnaker.h5'
if os.path.exists(filename):
    os.remove(filename)
h5py_wrapper.wrapper.add_to_h5(
    filename,
    cc['spinnaker'], write_mode='w', overwrite_dataset=True)

filename = '../../results/release_demo/viz_output_pop_spinnaker.pkl'
if os.path.exists(filename):
    os.remove(filename)
f = open(filename, 'w')
pickle.dump(cc['spinnaker'], f)
f.close()


filename = '../../results/release_demo/viz_output_pop_nest.h5'
if os.path.exists(filename):
    os.remove(filename)
h5py_wrapper.wrapper.add_to_h5(
    filename,
    cc['nest'], write_mode='w', overwrite_dataset=True)

filename = '../../results/release_demo/viz_output_pop_nest.pkl'
if os.path.exists(filename):
    os.remove(filename)
f = open(filename, 'w')
pickle.dump(cc['nest'], f)
f.close()
