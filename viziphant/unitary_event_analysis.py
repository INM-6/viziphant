
import math
import numpy
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import string
import neo
import elephant.unitary_event_analysis as ue

# ToDo: meaningful coloring?! ->all colorable_objects are now user changeable
# ToDo: Input Events as neo objects/ quantities
# ToDo: check user entries (=Benutzereingaben)
# ToDo: rearange the plotting parameters dict
# ToDo: panel sorting + selection -> modularisierung
# ToDo: use markerdict ->created, yet unreviewed
# ToDo: set trial labels
# ToDo: optional epochs/events + label
# ToDo: surprise representation??
# ToDo: Make relation between panels clearer?!
# ToDo: optional or even remove alphabetic labeling
# ToDo: improve neuron separation -> schwarzer trennungsstrich -> dicke verringert!

#plot_params_default = dictionary { keys : values }
plot_params_default = {
    # epochs to be marked on the time axis      #epochs = Def.: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    'events': [], #leere Liste
    # id of the units
    'unit_ids': [0, 1],
    # horizontal white space between subplots
    'hspace': 1,
    # width white space between subplots
    'wspace': 0.5,
    # font size         #Schriftgroesse
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': [1, 2],
    # line width
    'lw': 0.5,
    # y limit for the surprise
    'S_ylim': (-3, 3),

}

plot_markers_default = {
    'data_symbol': ".",
    'data_markersize': 0.5,
    #data_markercolor-tupel should be changed to a tupel with multiple elements,
    # if multiple colors are needed
    'data_markercolor': ("k"),
    'data_markerfacecolor': "none",
    'data_markeredgecolor': "none",
    'event_symbol': "s",
    'event_markersize': 5,
    'event_markercolor': "r",
    'event_markerfacecolor': "none",
    'event_markeredgecolor': "r",
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


def plot_unitary_event(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep, numberOfNeurons, plot_params_user,
            plot_markers_user, position):

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
        -plot_markers_user: list of dictionaries
            marker properties from the user
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
    print("classes of the parameters from 'plot_unitary_event':")
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
        raise ValueError('length of unit_ids should be equal to number of neurons! \n'
                         'Unit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    if 'suptitle' in plot_params.keys():                                                                                #'suptitle' im Default nicht vorhanden, kann also nur ueber plot_params_user eingepflegt werden
        plt.suptitle("Trial aligned on " +
                     plot_params['suptitle'], fontsize=20)
    plt.subplots_adjust(hspace=plot_params['hspace'], wspace=plot_params['wspace'])

    plot_spike_events(data, winsize, winstep, numberOfNeurons, plot_params_user, plot_markers_user[0],
                     position[0])

    plot_spike_rates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons,
                    plot_params_user, plot_markers_user[1], position[1])

    plot_coincidence_events(data, jointSuprise_dict, binsize, winsize, winstep, numberOfNeurons,
                           plot_params_user, plot_markers_user[2], position[2])

    plot_coincidence_rates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons,
                          plot_params_user,plot_markers_user[3], position[3])

    plot_statistical_significance(data, jointSuprise_dict, jointSuprise_sig, winsize, winstep,
                                 numberOfNeurons, plot_params_user, plot_markers_user[4], position[4])

    plot_UnitaryEvents(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep,
                       numberOfNeurons, plot_params_user, plot_markers_user[5], position[5])
    return None


def plot_spike_events(data, winsize, winstep, numberOfNeurons, plot_params_user, plot_markers_user, position):
    print('plotting Spike Events as raster plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # subplots format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax0 = plt.subplot(position[0], position[1], position[2])
    ax0.set_title('Spike Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax0.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *tr + n * (num_tr + 1) + 1,ls='none',
                     marker=plot_markers['data_symbol'], color=plot_markers['data_markercolor'][0],
                     markersize=plot_markers['data_markersize'])
        if n < numberOfNeurons - 1:
            ax0.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')

    ax0.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax0.set_ylim(0, (tr + 2) * (n + 1) + 1)

    ax0.xaxis.set_major_locator(MultipleLocator(200))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #set yaxis
    yticks_list = []
    for yt1 in range(1, numberOfNeurons*num_tr, num_tr+1):
        yticks_list.append(yt1)
    for n in range(numberOfNeurons):
        for yt2 in range(n*(num_tr+1)+15, (n+1)*num_tr, 15):
            yticks_list.append(yt2)
    yticks_list.sort()

    yticks_labels_list = [1]
    anzahl_y_ticks_proNeuron = math.floor(num_tr/15)
    for i in range(anzahl_y_ticks_proNeuron):
        yticks_labels_list.append((i+1)*15)

    hilfsListe = yticks_labels_list
    for i in range(numberOfNeurons-1):
        yticks_labels_list += hilfsListe
    #print(yticks_list)
    ax0.set_yticks(yticks_list)
    ax0.set_yticklabels(yticks_labels_list, fontsize=plot_params['fsize'])

    x_lim = ax0.get_xlim()
    ax0.text(x_lim[1], num_tr * 2 + 7, 'Neuron 2')
    ax0.text(x_lim[1], -12, 'Neuron 1')

    ax0.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax0.set_ylabel('Trial', fontsize=plot_params['fsize'])

    return None


def plot_spike_rates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons, plot_params_user,
                    plot_markers_user, position):
    print('plotting Spike Rates as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
                                                                                                     #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax1 = plt.subplot(position[0], position[1], position[2])
    ax1.set_title('Spike Rates')
    for n in range(numberOfNeurons):
        ax1.plot(t_winpos + winsize / 2.,
                 jointSuprise_dict['rate_avg'][:, n].rescale('Hz'),
                 label='Neuron ' + str(plot_params['unit_real_ids'][n]),
                 color=plot_markers['data_markercolor'][n], lw=plot_params['lw'])

    ax1.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    max_val_psth = 40
    ax1.set_ylim(0, max_val_psth)

    ax1.xaxis.set_major_locator(MultipleLocator(200))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax1.xaxis.set_minor_locator(MultipleLocator(100))
    ax1.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])

    ax1.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax1.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax1.set_ylabel('(1/s)', fontsize=plot_params['fsize'])

    return None


def plot_coincidence_events(data, jointSuprise_dict, binsize, winsize, winstep, numberOfNeurons,
                           plot_params_user, plot_markers_user, position):
    print('plotting Raw Coincidences as raster plot with markers indicating the Coincidences')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    ax2 = plt.subplot(position[0], position[1], position[2])
    ax2.set_title('Coincidence Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax2.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1, ls='None',
                     marker=plot_markers['data_symbol'], markersize=plot_markers['data_markersize'],
                     color=plot_markers['data_markercolor'])
            ax2.plot(numpy.unique(jointSuprise_dict['indices']['trial' + str(tr)]) *binsize,
                numpy.ones_like(numpy.unique(jointSuprise_dict['indices']['trial' + str(tr)]))
                * tr + n * (num_tr + 1) + 1,
                ls='', ms=plot_markers['event_markersize'], marker=plot_markers['event_symbol'],
                markerfacecolor=plot_markers['event_markerfacecolor'],
                markeredgecolor=plot_markers['event_markeredgecolor'])
        if n < numberOfNeurons - 1:
            ax2.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')
    ax2.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax2.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)

    ax2.xaxis.set_major_locator(MultipleLocator(200))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.xaxis.set_minor_locator(MultipleLocator(100))
    # set yaxis
    yticks_list = []
    for yt1 in range(1, numberOfNeurons * num_tr, num_tr + 1):
        yticks_list.append(yt1)
    for n in range(numberOfNeurons):
        for yt2 in range(n * (num_tr + 1) + 15, (n + 1) * num_tr, 15):
            yticks_list.append(yt2)
    yticks_list.sort()

    yticks_labels_list = [1]
    anzahl_y_ticks_proNeuron = math.floor(num_tr / 15)
    for i in range(anzahl_y_ticks_proNeuron):
        yticks_labels_list.append((i + 1) * 15)

    hilfsListe = yticks_labels_list
    for i in range(numberOfNeurons - 1):
        yticks_labels_list += hilfsListe
    # print(yticks_list)
    ax2.set_yticks(yticks_list)
    ax2.set_yticklabels(yticks_labels_list, fontsize=plot_params['fsize'])

    ax2.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax2.set_ylabel('Trial', fontsize=plot_params['fsize'])

    return None


def plot_coincidence_rates(data, jointSuprise_dict, winsize, winstep, numberOfNeurons, plot_params_user,
                          plot_markers_user, position):
    print('plotting empirical and expected coincidences rate as line plots')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \n'
                         'Unit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax3 = plt.subplot(position[0], position[1], position[2])
    ax3.set_title('Coincidence Rates')
    ax3.plot(t_winpos + winsize / 2.,
             jointSuprise_dict['n_emp'] / (winsize.rescale('s').magnitude * num_tr),
             label='empirical', lw=plot_params['lw'], color=plot_markers['data_markercolor'][0])
    ax3.plot(t_winpos + winsize / 2.,
             jointSuprise_dict['n_exp'] / (winsize.rescale('s').magnitude * num_tr),
             label='expected', lw=plot_params['lw'], color=plot_markers['data_markercolor'][1])
    ax3.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)

    ax3.xaxis.set_major_locator(MultipleLocator(200))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax3.xaxis.set_minor_locator(MultipleLocator(100))
    y_ticks = ax3.get_ylim()
    ax3.set_yticks([0, y_ticks[1] / 2, y_ticks[1]])

    ax3.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    ax3.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax3.set_ylabel('(1/s)', fontsize=plot_params['fsize'])

    return None


def plot_statistical_significance(data, jointSuprise_dict, jointSuprise_sig, winsize, winstep, numberOfNeurons,
                                 plot_params_user, plot_markers_user, position):
    print('plotting Surprise/Statistical Significance as line plot')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN

    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # figure format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \nUnit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax4 = plt.subplot(position[0], position[1], position[2])
    ax4.set_title('Statistical Significance')
    ax4.plot(t_winpos + winsize / 2., jointSuprise_dict['Js'], lw=plot_params['lw'],
             color=plot_markers['data_markercolor'][0])
    ax4.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax4.set_ylim(plot_params['S_ylim'])

    ax4.axhline(jointSuprise_sig, ls='-', color=plot_markers['data_markercolor'][1])
    ax4.axhline(-jointSuprise_sig, ls='-', color=plot_markers['data_markercolor'][2])
    ax4.text(t_winpos[30], jointSuprise_sig + 0.3, '$\\alpha +$', color=plot_markers['data_markercolor'][1])
    ax4.text(t_winpos[30], -jointSuprise_sig - 0.5, '$\\alpha -$', color=plot_markers['data_markercolor'][2])

    ax4.xaxis.set_major_locator(MultipleLocator(200))
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax4.xaxis.set_minor_locator(MultipleLocator(100))
    ax4.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])

    ax4.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax4.set_yticklabels([alpha+0.5, alpha, alpha-0.5])

    return None


def plot_unitary_event(data, jointSuprise_dict, jointSuprise_sig, binsize, winsize, winstep,
                       numberOfNeurons, plot_params_user, plot_markers_user, position):
    print('plotting Unitary Events as raster plot with markers indicating the Unitary Events')

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop
    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)                                                                                                              #jointSuprise_sig war NotANumber;; vorher jointSuprise_sig = ue.jointJ(sig_level) ,d.h. doppelter ue.jointJ aufruf und deshalb war jointSuprise_sig NAN
    num_tr = len(data)
    ls = '-'
    alpha = 0.5
                                                                                                                        #pat = ue.inverse_hash_from_pattern(pattern_hash, numberOfNeurons) # base fehlt?! ~/anaconda3/envs/vizitest/lib/python3.7/site-packages/elephant/unitary_event_analysis.py in inverse_hash_from_pattern(h, numberOfNeurons, base)
    # subplot format
    plot_params = plot_params_default.copy()
    plot_params.update(plot_params_user)
    # marker format
    plot_markers = plot_markers_default.copy()
    plot_markers.update(plot_markers_user)

    if len(plot_params['unit_real_ids']) != numberOfNeurons:
        raise ValueError('length of unit_ids should be equal to number of neurons! \n'
                         'Unit_Ids: '+plot_params['unit_real_ids'] +'ungleich NumOfNeurons: '+numberOfNeurons)
    plt.rcParams.update({'font.size': plot_params['fsize']})
    plt.rc('legend', fontsize=plot_params['fsize'])

    ax5 = plt.subplot(position[0], position[1], position[2])
    ax5.set_title('Unitary Events')
    for n in range(numberOfNeurons):
        for tr, data_tr in enumerate(data):
            ax5.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     ls='None', marker=plot_markers['data_symbol'],
                     markersize=plot_markers['data_markersize'], color=plot_markers['data_markercolor'])
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
                        ms=plot_markers['event_markersize'], marker=plot_markers['event_symbol'], ls='',
                        markerfacecolor=plot_markers['event_markerfacecolor'],
                        markeredgecolor=plot_markers['event_markeredgecolor'])

        if n < numberOfNeurons - 1:
            ax5.axhline((tr + 2) * (n + 1), lw=plot_params['lw'], color='b')
    ax5.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax5.set_ylim(0, (tr + 2) * (n + 1) + 1)

    x_line_coords = MultipleLocator(200).tick_values(0, 2200)
    for xc in x_line_coords:
        ax5.axvline(xc, lw=plot_params['lw'],color='g')

    ax5.xaxis.set_major_locator(MultipleLocator(200))
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax5.xaxis.set_minor_locator(MultipleLocator(100))
    # set yaxis
    yticks_list = []
    for yt1 in range(1, numberOfNeurons * num_tr, num_tr + 1):
        yticks_list.append(yt1)
    for n in range(numberOfNeurons):
        for yt2 in range(n * (num_tr + 1) + 15, (n + 1) * num_tr, 15):
            yticks_list.append(yt2)
    yticks_list.sort()

    yticks_labels_list = [1]
    anzahl_y_ticks_proNeuron = math.floor(num_tr / 15)
    for i in range(anzahl_y_ticks_proNeuron):
        yticks_labels_list.append((i + 1) * 15)

    hilfsListe = yticks_labels_list
    for i in range(numberOfNeurons - 1):
        yticks_labels_list += hilfsListe
    # print(yticks_list)
    ax5.set_yticks(yticks_list)
    ax5.set_yticklabels(yticks_labels_list, fontsize=plot_params['fsize'])

    ax5.set_xlabel('Time [ms]', fontsize=plot_params['fsize'])
    ax5.set_ylabel('Trial', fontsize=plot_params['fsize'])

    return None


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
"""