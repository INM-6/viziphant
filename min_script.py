from neo.core import SpikeTrain
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from elephant.spike_train_generation import homogeneous_gamma_process as HGP
from quantities import s, Hz
import imp
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '/home/robin/Projects/INM6/Tasks/viziphant/'
plotting_path = './plots/generic.py'
generic = imp.load_source('*', plotting_path)


st_list1 = [HPP(rate=10*Hz) for _ in range(100)]
st_list2 = [HGP(3, 10*Hz) for _ in range(100)]

# plot visually separates the two lists
generic.rasterplot([st_list1, st_list2])

# Add annotations to spiketrains
for i, (st1, st2) in enumerate(zip(st_list1, st_list2)):
    if i.__mod__(2):
        st1.annotations['parity'] = 'odd'
        st2.annotations['parity'] = 'odd'
    else:
        st1.annotations['parity'] = 'even'
        st2.annotations['parity'] = 'even'

# plot separates the list and the annotation values within each list
generic.rasterplot([st_list1, st_list2], key_list=['parity'],
                   groupingdepth=2, labelkey='0+1')

# '' key can change the priority of the list grouping
generic.rasterplot([st_list1, st_list2], key_list=['parity', ''],
                   groupingdepth=2, labelkey='0+1')

# groups can also be emphasized by an explicit color code
ax, axhistx, axhisty = generic.rasterplot([st_list1, st_list2], key_list=['', 'parity'],
                   groupingdepth=1, labelkey=0, colorkey='parity', legend=True)
axhisty.set_xlabel('######')

plt.show()



