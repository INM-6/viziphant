

import numpy
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import string
import neo
import elephant.unitary_event_analysis as ue

# ToDo: meaningful coloring?!
# ToDo: Input Events as neo objects/ quantities
# ToDo: check user entries (=Benutzereingaben)
# ToDo: rearange the plotting parameters dict
# ToDo: panel sorting + selection -> modularisierung
# ToDo: use markerdict
# ToDo: set trial labels
# ToDo: optional epochs/events + label
# ToDo: surprise representation??
# ToDo: Make relation between panels clearer?!
# ToDo: optional or even remove alphabetic labeling
# ToDo: improve neuron separation -> schwarzer trennungsstrich

#plot_params_default = dictionary { keys : values }
plot_params_default = {
    # epochs to be marked on the time axis      #epochs = Def.: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    'events': [], #leere Liste
    # save figure
    'save_fig': False,
    # show figure
    'showfig': True,
    # figure size
    'figsize': (10, 12), #Tupel # entfernen
    # right margin
    'right': 0.9,  # entfernen
    # top margin
    'top': 0.9,  # entfernen
    # bottom margin
    'bottom': 0.1, # entfernen
    # left margin
    'left': 0.1, # entfernen
    # id of the units
    'unit_ids': [0, 1],
    # delete the x ticks when "False"   #ticks ->marker on the axis
    'set_xticks': False,
    # horizontal white space between subplots
    'hspace': 0.5,
    # width white space between subplots
    'wspace': 0.5,
    # font size         #Schriftgroesse
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': [1, 2],
    # channel id
    'ch_real_ids': [],
    # line width
    'lw': 2,        #gleich zeile 47 'linewidth' :2
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # marker size for the UEs and coincidences
    'ms': 5,
    # path and file name for saving the figure
    'path_filename_format': 'figure.pdf'
}


def load_gdf2Neo(fname, trigger, t_pre, t_post):
    """
    load and convert the gdf file to Neo format by
    cutting and aligning around a given trigger
    # codes for trigger events (extracted from a
    # documentation of an old file after
    # contacting Dr. Alexa Riehle)
    # 700 : ST (correct) 701, 702, 703, 704*
    # 500 : ST (error =5) 501, 502, 503, 504*
    # 1000: ST (if no selec) 1001,1002,1003,1004*
    # 11  : PS 111, 112, 113, 114
    # 12  : RS 121, 122, 123, 124
    # 13  : RT 131, 132, 133, 134
    # 14  : MT 141, 142, 143, 144
    # 15  : ES 151, 152, 153, 154
    # 16  : ES 161, 162, 163, 164
    # 17  : ES 171, 172, 173, 174
    # 19  : RW 191, 192, 193, 194
    # 20  : ET 201, 202, 203, 204

    Parameters:
        -fname: String
            name of the gdf-file
        -trigger: String
            spezification of trigger-kind
        -t_pre: number (int or float?)
            inicial time
        -t_post: number (int or float?)
            final time

    Returns:
        -spiketrain: 1D-Array
            spiketrains as representation of neural activity
    """

    print("classes of the parameters from 'load_gdf2NEO':")
    print(type(fname))
    print(type(trigger))
    print(type(t_pre))
    print(type(t_post))

    try:
        _checkingUserEntries_load_gdf2Neo(fname, trigger, t_pre, t_post)
    except TypeError as TE:
        print(TE)
        raise TE


    data = numpy.loadtxt(fname)         #ist data ein 2D-Feld?? -> (Ja), numpy.loadtxt erstellt abhaengig vom uebergebenen Input immer ein array, dimensonen sind dabei variabel
    if trigger == 'PS_4':
        trigger_code = 114
    if trigger == 'RS_4':
        trigger_code = 124
    if trigger == 'RS':
        trigger_code = 12
    if trigger == 'ES':
        trigger_code = 15
    # specify units
    units_id = numpy.unique(data[:, 0][data[:, 0] < 7])
    # indecies of the trigger
    sel_tr_idx = numpy.where(data[:, 0] == trigger_code)[0] #selectTriggerIndex
    # cutting the data by aligning on the trigger
    data_tr = []                #leereListe fuer DataTrigger
    for id_tmp in units_id:
        data_sel_units = []     #fuer jede unit in data wird eine leereUnitListe erstellt
        for i_cnt, i in enumerate(sel_tr_idx):
            start_tmp = data[i][1] - t_pre.magnitude
            stop_tmp = data[i][1] + t_post.magnitude
            sel_data_tmp = numpy.array(
                data[numpy.where((data[:, 1] <= stop_tmp) &
                                 (data[:, 1] >= start_tmp))])
            sp_units_tmp = sel_data_tmp[:, 1][
                numpy.where(sel_data_tmp[:, 0] == id_tmp)[0]]
            if len(sp_units_tmp) > 0:                           #laenge von 2D-Feld 'sp_units_temp' entspricht Anzahl der der Elemente inm array
                aligned_time = sp_units_tmp - start_tmp
                data_sel_units.append(neo.SpikeTrain(
                    aligned_time * pq.ms, t_start=0 * pq.ms,
                    t_stop=t_pre + t_post))
            else:
                data_sel_units.append(neo.SpikeTrain(
                    [] * pq.ms, t_start=0 * pq.ms,
                    t_stop=t_pre + t_post))
        data_tr.append(data_sel_units)
    data_tr.reverse()                                            #warum einmal reversen???
    spiketrain = numpy.vstack([i for i in data_tr]).T            #numpy.vstack: Stack arrays in sequence vertically
    return spiketrain


def plot_UE(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep, numberOfNeurons, plot_params_user,
            position):

    """
    Visualization of the results of the Unitary Event Analysis.

    Parameters:
        -data: list of spiketrains
            list of spike trains in different trials as representation of neural activity
        -jointSuprise_dict: dictionary
            JointSuprise dictionary
        -jointSuprise_sig: list of floats
            list of suprise measure
        -binsize: Quantity scalar with dimension time
           size of bins for descritizing spike trains
        -winsize: Quantity scalar with dimension time
           size of the window of analysis
        -winstep: Quantity scalar with dimension time
           size of the window step
        -pattern_hash: list of integers
           list of interested patterns in hash values
           (see hash_from_pattern and inverse_hash_from_pattern functions)
        -numberOfNeurons: integer
            number of Neurons
        -plot_params_user: dictionary
            plotting parameters from the user
        -position: list of position-tupels
            (posSpikeEvents(c,r,i), posSpikeRates(c,r,i), posCoincidenceEvents(c,r,i), pos CoincidenceRates(c,r,i),
            posStatisticalSignificance(c,r,i), posUnitaryEvents(c,r,i))
            pos is a three integer-tupel, where the first integer is the number of rows,
            the second the number of columns, and the third the index of the subplot

    Returns:
        -NONE

    The following plots will be created:
    - Spike Events (as rasterplot)
    - Spike Rates (as curve)
    - Coincident Events (as rasterplot with markers)
    - Empirical & Excpected Coincidences Rates (as curves)
    - Suprise or Statistical Significance (as curve with alpha-limits)
    - Unitary Events (as rasterplot with markers)


    Unitary Event (UE) analysis is a statistical method that
     enables to analyze in a time resolved manner excess spike correlation
     between simultaneously recorded neurons by comparing the empirical
     spike coincidences (precision of a few ms) to the expected number
     based on the firing rates of the neurons.
    References:
      - Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
        94(1): 67-79.
      - Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
      - Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
        nonstationarity on joint-spike events Biological Cybernetics 88(5):335-351.
      - Gruen S (2009) Data-driven significance estimation of precise spike
        correlation. J Neurophysiology 101:1126-1140 (invited review)
    :copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
    :license: Modified BSD, see LICENSE.txt for details.
    """
    """
    print("classes of the parameters from 'plot_UE':")
    print("data: "+str(type(data)))
    print("jointSuprise_dict: "+str(type(jointSuprise_dict)))
    print("jointSuprise_sig: "+str(type(jointSuprise_sig)))
    print("binsize: "+str(type(binsize)))
    print("winsize: "+str(type(winsize)))
    print("winstep: "+str(type(winstep)))
    print("pattern_hash: "+str(type(pattern_hash)))
    print("numberOfNeurons: "+str(type(numberOfNeurons)))
    print("plot_params_user: "+str(type(plot_params_user))+"\n")
    """
    """
    try:
        _checkungUserEntries_plot_UE(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep, numberOfNeurons, plot_params_user)
    except (TypeError, KeyError) as errors:
        print(errors)
        raise errors
    """
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \nUnit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    if 'suptitle' in plot_params.keys():                                                                                #'suptitle' im Default nicht vorhanden, kann also nur ueber plot_params_user eingepflegt werden
        plt.suptitle("Trial aligned on " +
                     plot_params['suptitle'], fontsize=20)
    plt.subplots_adjust(top=plot_params.get('top'), right=plot_params['right'], left=plot_params['left'],
                        bottom=plot_params['bottom'], hspace=plot_params['hspace'], wspace=plot_params['wspace'])


    plot_SpikeEvents(data, winsize, winstep, numberOfNeurons, plot_params_user, position[0])

    plot_SpikeRates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons, plot_params_user, position[1])

    plot_CoincidenceEvents(data, jointSuprise_dict, binsize, winsize, winstep, numberOfNeurons,
                           plot_params_user, position[2])

    plot_CoincidenceRates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons,
                          plot_params_user, position[3])

    plot_StatisticalSignificance(data, jointSuprise_dict, jointSuprise_sig, winsize, winstep, numberOfNeurons,
                                 plot_params_user, position[4])

    plot_UnitaryEvents(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep, numberOfNeurons,
                       plot_params_user, position[5])
    return None


def plot_SpikeEvents(data, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting Spike Events as raster plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    ax0 = plt.subplot(position[0], position[1], position[2])
    ax0.set_title('Spike Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax0.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     '.', markersize=0.5, color='k')
        if n < numberOfNeurons - 1:
            ax0.axhline((tr + 2) * (n + 1), lw=2,
                        color='k')  # deadCode: default: lw = 2; ->Nein, da lw von plt.rc kommt
    ax0.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax0.set_yticks([num_tr + 1, num_tr + 16, num_tr + 31])
    ax0.set_yticklabels([1, 15, 30], fontsize=plot_params['fsize'])
    ax0.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax0.set_xticks([])
    ax0.set_ylabel('Trial', fontsize=plot_params['fsize'])
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax0.axvline(e_val, ls=ls, color='r', lw=2,
                        alpha=alpha)  # deadCode: default: lw = 2;  ->Nein, da lw von plt.rc kommt
    Xlim = ax0.get_xlim()
    ax0.text(Xlim[1], num_tr * 2 + 7, 'Neuron 1')
    ax0.text(Xlim[1], -12, 'Neuron 2')


def plot_SpikeRates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting Spike Rates as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    ax1 = plt.subplot(position[0], position[1], position[2])
    ax1.set_title('Spike Rates')
    for n in range(numberOfNeurons):
        ax1.plot(t_winpos + winsize / 2.,
                 jointSuprise_dict['rate_avg'][:, n].rescale('Hz'),
                 label='Neuron ' + str(plot_params['unit_real_ids'][n]), lw=plot_params['lw'])
    ax1.set_ylabel('(1/s)', fontsize=plot_params['fsize'])
    ax1.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    max_val_psth = 40
    ax1.set_ylim(0, max_val_psth)
    ax1.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])
    ax1.legend(
        bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax1.axvline(e_val, ls=ls, color='r', lw=plot_params['lw'], alpha=alpha)
    ax1.set_xticks([])


def plot_CoincidenceEvents(data, jointSuprise_dict, binsize, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting Raw Coincidences as raster plot with markers indicating the Coincidences')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    ax2 = plt.subplot(position[0], position[1], position[2])
    ax2.set_title('Coincidence Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax2.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     '.', markersize=0.5, color='k')
            ax2.plot(
                numpy.unique(jointSuprise_dict['indices']['trial' + str(tr)]) *
                binsize,
                numpy.ones_like(numpy.unique(jointSuprise_dict['indices'][
                                                 'trial' + str(tr)])) * tr + n * (num_tr + 1) + 1,
                ls='', ms=plot_params['ms'], marker='s', markerfacecolor='none',
                markeredgecolor='c')
        if n < numberOfNeurons - 1:
            ax2.axhline((tr + 2) * (n + 1), lw=2, color='k')
    ax2.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax2.set_yticks([num_tr + 1, num_tr + 16, num_tr + 31])
    ax2.set_yticklabels([1, 15, 30], fontsize=plot_params['fsize'])
    ax2.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax2.set_xticks([])
    ax2.set_ylabel('Trial', fontsize=plot_params['fsize'])
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax2.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)


def plot_CoincidenceRates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting empirical and expected coincidences rate as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \nUnit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax3 = plt.subplot(position[0], position[1], position[2])
    ax3.set_title('Coincidence Rates')
    ax3.plot(t_winpos + winsize / 2.,
             jointSuprise_dict['n_emp'] / (winsize.rescale('s').magnitude * num_tr),
             label='empirical', lw=plot_params['lw'], color='c')
    ax3.plot(t_winpos + winsize / 2.,
             jointSuprise_dict['n_exp'] / (winsize.rescale('s').magnitude * num_tr),
             label='expected', lw=plot_params['lw'], color='m')
    ax3.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax3.set_ylabel('(1/s)', fontsize=plot_params['fsize'])
    ax3.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    YTicks = ax3.get_ylim()
    ax3.set_yticks([0, YTicks[1] / 2, YTicks[1]])
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax3.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)
    ax3.set_xticks([])

def plot_StatisticalSignificance(data, jointSuprise_dict, jointSuprise_sig, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting Surprise/Statistical Significance as line plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \nUnit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax4 = plt.subplot(position[0], position[1], position[2])
    ax4.set_title('Statistical Significance')
    ax4.plot(t_winpos + winsize / 2., jointSuprise_dict['Js'], lw=plot_params['lw'], color='k')
    ax4.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax4.axhline(jointSuprise_sig, ls='-', color='r')
    ax4.axhline(-jointSuprise_sig, ls='-', color='g')
    ax4.text(t_winpos[30], jointSuprise_sig + 0.3, '$\\alpha +$', color='r')
    ax4.text(t_winpos[30], -jointSuprise_sig - 0.5, '$\\alpha -$', color='g')
    ax4.set_xticks(t_winpos.magnitude[::int(len(t_winpos) / 10)])
    ax4.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])
    ax4.set_yticklabels([0.99, 0.5, 0.01])

    ax4.set_ylim(plot_params['S_ylim'])
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax4.axvline(e_val, ls=ls, color='r', lw=plot_params['lw'], alpha=alpha)
    ax4.set_xticks([])

def plot_UnitaryEvents(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep, numberOfNeurons, plot_params_user, position):
    print('plotting Unitary Events as raster plot with markers indicating the Unitary Events')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \nUnit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax5 = plt.subplot(position[0], position[1], position[2])
    ax5.set_title('Unitary Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax5.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1, '.',
                     markersize=0.5, color='k')
            sig_idx_win = numpy.where(jointSuprise_dict['Js'] >= jointSuprise_sig)[0]
            if len(sig_idx_win) > 0:
                x = numpy.unique(jointSuprise_dict['indices']['trial' + str(tr)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = numpy.append(xx, x[numpy.where(
                            (x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + winsize))])
                    ax5.plot(
                        numpy.unique(xx) * binsize,
                        numpy.ones_like(numpy.unique(xx)) * tr + n * (num_tr + 1) + 1,
                        ms=plot_params['ms'], marker='s', ls='', markerfacecolor='none', markeredgecolor='r')

        if n < numberOfNeurons - 1:
            ax5.axhline((tr + 2) * (n + 1), lw=2, color='k')
    ax5.set_yticks([num_tr + 1, num_tr + 16, num_tr + 31])
    ax5.set_yticklabels([1, 15, 30], fontsize=plot_params['fsize'])

    ax5.xaxis.set_major_locator(MultipleLocator(200))
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax5.xaxis.set_minor_locator(MultipleLocator(100))

    ax5.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax5.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)

    ax5.set_ylabel('Trial', fontsize=plot_params['fsize'])
    ax5.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    for key in plot_params['events']:
        for e_val in plot_params['events'][key]:
            ax5.axvline(e_val, ls=ls, color='r', lw=2, alpha=alpha)
            ax5.text(e_val - 10 * pq.ms,
                     plot_params['S_ylim'][0] - 35, key, fontsize=plot_params['fsize'], color='r')

"""
def _checkingUserEntries_load_gdf2Neo(fname, trigger, t_pre, t_post):
    if (type(fname) != str):
        raise TypeError('fname must be a string')
    if (type(trigger) != str):
        raise TypeError('trigger must be a string')
    if (type(t_pre) != float): # or int    ## negative values allowed?
        raise TypeError('t_pre must be a float/integer')
    if (type(t_post) != float):
        raise TypeError('t_post must be a float/integer')

def _checkungUserEntries_plot_UE(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep,
            pattern_hash, numberOfNeurons, plot_params_user):
    if (type(data) != list and type(data) != numpy.ndarray): # sollen weiter Typen erlaubt sein???
        raise TypeError('data must be a list (of spiketrains)')

    if (type(jointSuprise_dict) != dict):
        raise TypeError('jointSuprise_dict must be a dictionary')
    else: #checking if all keys are correct
        if "Js" not in jointSuprise_dict:
            raise KeyError('"Js"-key is missing in jointSuprise_dict')
        if "indices" not in jointSuprise_dict:
            raise KeyError('"indices"-key is missing in jointSuprise_dict')
        if "n_emp" not in jointSuprise_dict:
            raise KeyError('"n_emp"-key is missing in jointSuprise_dict')
        if "n_exp" not in jointSuprise_dict:
            raise KeyError('"n_exp"-key is missing in jointSuprise_dict')
        if "rate_avg" not in jointSuprise_dict:
            raise KeyError('"rate_avg"-key is missing in jointSuprise_dict')
        #creating keys-list and removing all legal keys
        keys_jointSuprise_dict = list(jointSuprise_dict.keys())
        keys_jointSuprise_dict.remove("Js")
        keys_jointSuprise_dict.remove("indices")
        keys_jointSuprise_dict.remove("n_emp")
        keys_jointSuprise_dict.remove("n_exp")
        keys_jointSuprise_dict.remove("rate_avg")
        if (len(keys_jointSuprise_dict) != 0): # checking for additional invalid keys
            raise KeyError('invalid keys in jointSuprise_dict detected')

    if ( (type(jointSuprise_sig) != list) and (type(jointSuprise_sig) != numpy.float64)
            and (type(jointSuprise_sig) != numpy.ndarray)  ):
        raise TypeError('jointSuprise_sig must be a list (of floats)')
    elif (type(jointSuprise_sig) == list):
        for i in jointSuprise_sig:
            if ( (type(jointSuprise_sig[i]) != numpy.float64) and (type(jointSuprise_sig[i]) != float) ):
                raise TypeError('elements of the jointSuprise_sig list are NOT floats')

    if (type(binsize) != pq.quantity.Quantity): #quantity scaler
        raise TypeError('binsize must be a quantity scaler/int')

    if (type(winsize) != pq.quantity.Quantity):
        raise TypeError('winsize must be a quantity scaler/int')

    if (type(winstep)!= pq.quantity.Quantity):
        raise TypeError('winstep must be a quantity scaler/int')

    if (type(pattern_hash) != list and type(pattern_hash) != numpy.ndarray):
        raise TypeError('pattern_hash must be a list (of integers)')
    elif (type(pattern_hash) == list):
        for i in pattern_hash:
            if (type(pattern_hash[i]) != int):
                raise TypeError('elements of the pattern_hash list are NOT integers')

    if (type(numberOfNeurons) != int):
        raise TypeError('numberOfNeurons must be an integer')

    if (type(plot_params_user) != dict):
        raise TypeError('plot_params_user must be a dictionary')
    else: #checking if all key are correct  ->not every key from the default-dict must be in the users-dict, but also no additional keys
        keys_plot_params_user = list(plot_params_user.keys())
        for x in list(plot_params_default.keys()):
            if x in keys_plot_params_user:
                keys_plot_params_user.remove(x)
        if (len(keys_plot_params_user) != 0):
            raise KeyError('invalid keys in plot_params_user detected')
"""