'''
This script collects the results of individual runs of hbp_review_task (run
either on UP or locally), and assembles a single, compressed result file (with
some additional info).
'''

# =============================================================================
# Initialization
# =============================================================================

# paths
import sys

# to find our "special" elephant
sys.path.insert(1, '..')
# change this to point to your reachgrasp IO
sys.path.insert(1, '../../dataset_repos/reachgrasp/python')
sys.path.insert(1, '../../toolboxes/py/python-neo')
sys.path.insert(1, '../../toolboxes/py/python-odml')
sys.path.insert(1, '../../toolboxes/py/csn_toolbox')

import os
import glob
import pickle

import numpy as np
import quantities as pq

# provides neo framework and I/Os to load exp and mdl data
import neo

# provides core analysis library component
import elephant

import h5py_wrapper.wrapper


def cch_measure(cch_all_pairs, times):
    ind = np.argmin(np.abs(times))
    return np.sum(cch_all_pairs[ind - 5:ind + 5])

# =============================================================================
# Global variables
# =============================================================================

# duration of recording to load
rec_start = 10.*pq.s
duration = 50.*pq.s


# =============================================================================
# Load experimental data
# =============================================================================

filename = 'data/experiment.h5'
session = neo.NeoHdf5IO(filename=filename)
block = session.read_block()

# select spike trains
sts_exp = block.filter(use_st=True)

print("Number of experimental spike trains: " + str(len(sts_exp)))

# create binned spike trains
sts_exp_bin = elephant.conversion.Binned(
    sts_exp, binsize=20 * pq.ms,
    t_start=rec_start, t_stop=rec_start + duration)

num_neurons = len(sts_exp)

# =============================================================================
# Load simulation data
# =============================================================================

filename = 'data/model.h5'
session = neo.NeoHdf5IO(filename=filename)
block = session.read_block()

# select spike trains
sts_mdl = block.filter(use_st=True)

print("Number of model spike trains: " + str(len(sts_mdl)))

# create binned spike trains
sts_mdl_bin = elephant.conversion.Binned(
    sts_mdl, binsize=20 * pq.ms,
    t_start=rec_start, t_stop=rec_start + duration)


# =============================================================================
# Calculate measures
# =============================================================================

rates = {}
rates['exp'] = [
    elephant.statistics.mean_firing_rate(st).rescale("Hz").magnitude
    for st in sts_exp]
rates['mdl'] = [
    elephant.statistics.mean_firing_rate(st).rescale("Hz").magnitude
    for st in sts_mdl]

isis_exp = [elephant.statistics.isi(st) for st in sts_exp]
isis_mdl = [elephant.statistics.isi(st) for st in sts_mdl]

cvs = {}
cvs['exp'] = [elephant.statistics.cv(isi) for isi in isis_exp]
cvs['mdl'] = [elephant.statistics.cv(isi) for isi in isis_mdl]

lvs = {}
lvs['exp'] = [elephant.statistics.lv(isi) for isi in isis_exp]
lvs['mdl'] = [elephant.statistics.lv(isi) for isi in isis_mdl]


# =============================================================================
# Rewrite files
# =============================================================================

num_edges = 0
for ni in range(num_neurons):
    for nj in range(ni, num_neurons):
        num_edges += 1

cc = {}
for dta in ['exp', 'mdl']:
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

# values per neuron
for dta, sts in zip(['exp', 'mdl'], [sts_exp, sts_mdl]):
    for neuron_i in range(num_neurons):
        channel = sts[neuron_i].unit.channel_indexes
        if type(channel) not in [int, float] and channel is not None:
            channel = channel[0]
        lin_channel = sts[neuron_i].unit.annotations['ca_id']
        cc[dta]['neuron_topo']['x'][neuron_i] = \
            int(lin_channel) / 10
        cc[dta]['neuron_topo']['y'][neuron_i] = \
            int(lin_channel) % 10

        if dta == 'exp':
            cc[dta]['neuron_single_values']['behavior'][neuron_i] = np.array([
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 0, 0, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 0, 0, 0, 1, 1, 1, 1,
                2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 0, 0, 3, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 3, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3])[neuron_i]

    cc[dta]['neuron_single_values']['rate'] = rates[dta]
    cc[dta]['neuron_single_values']['cv'] = cvs[dta]
    cc[dta]['neuron_single_values']['lv'] = lvs[dta]

# values per edge
num_tasks = len(glob.glob(
    '../results/hbp_review_task/correlation_output_*.h5'))
for job_parameter in range(num_tasks):
    filename = \
        '../results/hbp_review_task/correlation_output_' + \
        str(job_parameter) + '.h5'
    if not os.path.exists(filename):
        raise IOError('Cannot find file %s.', filename)
    print("Assembly of : %s" % filename)

    cc_part = h5py_wrapper.wrapper.load_h5(filename)

    for dta, sts in zip(['exp', 'mdl'], [sts_exp, sts_mdl]):
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
                cc_part[dta]['original'][calc_i]
            cc[dta]['edge_time_series']['sig_upper_975'][calc_i, :] = \
                cc_part[dta]['surr'][calc_i][smas[975], :]
            cc[dta]['edge_time_series']['sig_lower_25'][calc_i, :] = \
                cc_part[dta]['surr'][calc_i][smas[25], :]
            cc[dta]['edge_time_series']['times_ms'][calc_i, :] = \
                cc_part[dta]['times_ms'][calc_i]

    del cc_part

# write parameters to disk
filename = '../results/hbp_review_task/viz_output_exp.h5'
if os.path.exists(filename):
    os.remove(filename)
h5py_wrapper.wrapper.add_to_h5(
    filename,
    cc['exp'], write_mode='w', overwrite_dataset=True)

filename = '../results/hbp_review_task/viz_output_exp.pkl'
if os.path.exists(filename):
    os.remove(filename)
f = open(filename, 'w')
pickle.dump(cc['exp'], f)
f.close()


filename = '../results/hbp_review_task/viz_output_mdl.h5'
if os.path.exists(filename):
    os.remove(filename)
h5py_wrapper.wrapper.add_to_h5(
    filename,
    cc['mdl'], write_mode='w', overwrite_dataset=True)

filename = '../results/hbp_review_task/viz_output_mdl.pkl'
if os.path.exists(filename):
    os.remove(filename)
f = open(filename, 'w')
pickle.dump(cc['mdl'], f)
f.close()
