
import numpy
import quantities as pq
import elephant.unitary_event_analysis as ue
import neo
import matplotlib.pyplot as plt
from viziphant.unitary_event_analysis import (load_gdf2Neo, plot_UE, plot_params_default)


# parameters for unitary events analysis
winsize = 100 * pq.ms
binsize = 5 * pq.ms
winstep = 5 * pq.ms
pattern_hash = [3]
method = 'analytic_TrialAverage'
significance_level = 0.05
data_path = './'

# Figure 1 (alignment of the trials on the PS)

# This is the first attempt to reproduce Figure 2
# of the original article after cutting the data
# with the trial alignment on the PS.

# load and cut the data
file_name = 'winny131_23.gdf'

trigger = 'PS_4'
t_pre = 300 * pq.ms
t_post = 1800 * pq.ms

spiketrain = load_gdf2Neo(data_path + file_name, trigger, t_pre, t_post)
N = len(spiketrain.T)

# calculating UE ...
UE = ue.jointJ_window_analysis(
    spiketrain, binsize, winsize, winstep, pattern_hash, method=method)

# parameters for plotting
plot_params = {
    'events': {'0\nPS': [300 * pq.ms], '600\nES1': [900 * pq.ms],
               'ES2': [1200 * pq.ms], 'ES3': [1500 * pq.ms],
               '1500\nRS': [1800 * pq.ms]},
    'path_filename_format': '/figure1.eps',
}

print('plotting Figure 1 with trigger: ', trigger, '...')
plot_UE(spiketrain, UE, significance_level, binsize,
               winsize, winstep, pattern_hash, N, plot_params)


