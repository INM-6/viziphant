from quantities import Hz, ms
import numpy as np
import sys
import neo
import imp
import elephant.unitary_event_analysis as ue
from elephant.statistics import *
import matplotlib.pyplot as plt

plotting_path = './plots/generic.py'
plotting = imp.load_source('*', plotting_path)

elph_funtions = imp.load_source('*', './plots/elph_functions.py')

gen_data_path = '/home/robin/Projects/ValidationTools/validation/test_data.py'
test_data = imp.load_source('*', gen_data_path)

data = []

for trial in range(10):
    st_list = test_data.test_data(size=2,
                                  corr=.1,
                                  t_stop=10000*ms,
                                  rate=10*Hz,
                                  assembly_sizes=[2],
                                  method="CPP",
                                  bkgr_corr=.00)
    data += [st_list]


# plotting.rasterplot(data)
# plt.show()

Js_dict = ue.jointJ_window_analysis(data=data,
                                    binsize=5*ms,
                                    winsize=100*ms,
                                    winstep=20*ms,
                                    pattern_hash=[ue.hash_from_pattern([1, 1], 2)]
                                    )

print Js_dict

# Why not save settings in Js_dict?

dict_args = {'events': {'SO': [10 * ms]},
             'save_fig': True,
             'path_filename_format': 'UE1.pdf',
             'showfig': True,
             'suptitle': True,
             'figsize': (25, 20),
             'unit_ids': [10, 19],
             'ch_ids': [1, 3],
             'fontsize': 15,
             'linewidth': 2,
             'set_xticks': False,
             'marker_size': 8}

elph_funtions.plot_UE(data=data,
                      Js_dict=Js_dict,
                      sig_level=0.10,
                      binsize=2*ms,
                      winsize=100*ms,
                      winstep=20*ms,
                      pattern_hash=[ue.hash_from_pattern([1, 1], 2)],
                      N=2,
                      args=dict_args)

