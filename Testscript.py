import quantities as pq
import numpy as np
import sys
import neo
import imp
from elephant.statistics import *
# from elephant.spike_train_correlation import corrcoef
# from elephant.conversion import BinnedSpikeTrain
import matplotlib.pyplot as plt


# Define path names
DATA_PATH = '/home/robin/Projects/INM6/Tasks/viziphant/'
plotting_path = './plots/generic.py'
plotting = imp.load_source('*', plotting_path)

def load_data(path, file_name_list):
    # Load NEST or SpiNNaker data using NeoHdf5IO
    data = np.empty(file_name_list.size, dtype=object)
    for i, file_name in enumerate(file_name_list):
        # exc. and inh. as tuples, layerwise
        file_path = path + file_name
        data[i] = neo.io.NeoHdf5IO(file_path)
    return data

files = np.array(['spikes_L4E.h5'])
neo_obj = load_data(DATA_PATH, files)[0]
spiketrain_list = neo_obj.read_block().segments[0].spiketrains

fig = plt.figure('Rasterplot')
ax = fig.add_subplot(1,1,1)

N = spiketrain_list.__len__()

for i in range(N):
    if i.__mod__(2):
        spiketrain_list[i].annotations['key1'] = 'odd'
    else:
        spiketrain_list[i].annotations['key1'] = 'even'

def exclude_function(st):
    if st.size < 100:
        return True
    else:
        return False

def func(st):
    return 4

Q = N/4
ax, axhistx, axhisty = plotting.rasterplot([spiketrain_list[:1*Q], spiketrain_list[1*Q:2*Q]],# spiketrain_list[3*Q:]],
                               key_list=['key1'],
                               groupingdepth=2,
                               spacing=[7,2],
                               colorkey=0,
                               pophist_mode='color',
                               pophistbins=100,
                               right_histogram=mean_firing_rate,
                               filter_function=None,
                               histscale=.1,
                               labelkey='0+1',
                               markerargs={'markersize': 4, 'marker': '.'},
                               seperatorargs={'linewidth': 1, 'linestyle': '--', 'color': 'grey'},
                               legend=False,
                               legendargs={'loc': (1., 1.), 'markerscale': 1.5, 'handletextpad': 0},
                               ax=None,
                               style='ticks',
                               palette='Set2',
                               context='paper',
                               colorcodes='colorblind')

# axhistx.set_ylim(0,100)
plt.show()

