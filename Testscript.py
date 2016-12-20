import quantities as pq
import numpy as np
import sys
import neo
import elephant
import imp
from elephant.statistics import time_histogram, mean_firing_rate, cv, isi
from elephant.spike_train_correlation import corrcoef
from elephant.conversion import BinnedSpikeTrain
import matplotlib.pyplot as plt

# Define path names
DATA_PATH = '/home/robin/INM6/Projects/viziphant/'
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

fig = plt.figure('Rasterplot')
ax = fig.add_subplot(1,1,1)

plotting.rasterplot(ax, neo_obj)
plt.show(fig)