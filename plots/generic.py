"""
ToDo: Annotation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input as (list of) list of spiketrains
# Priorisierte Strukturierung nach Listen
# sort key Parameter der nach (liste von) Annotation sortiert
# analyse funktionen mit numpy/ elephant

def get_attributes(sorted_spiketrains, key_list, FullMatrix='False'):
    # find minimal grouping key:
    groupsizes = [0]
    key_count = len(key_list)
    attribute_array = np.zeros((sorted_spiketrains.__len__(), len(key_list)))
    group_key_nbr = 0
    if FullMatrix:
        group_bound = np.inf
    else:
        group_bound = 2
    while max(groupsizes) < group_bound and key_count > 0:
        # for full attribute array with value indicator for each key
        # remove max(groupsizes) < 2
        key_count -= 1
        group_key = key_list[key_count]
        groupsizes = [0]
        i = 0
        ref = 0
        values = np.array([])
        # count all group sizes for key:
        while i < sorted_spiketrains.__len__():
            if not np.where(values == sorted_spiketrains[i].annotations[group_key]):
                values = np.append(values, sorted_spiketrains[i].annotations[group_key])
                print 'add value'
                # ToDo: Debug! Does not include new values!
            # count group size for key:
            while sorted_spiketrains[i].annotations[group_key] == sorted_spiketrains[ref].annotations[group_key]:
                attribute_array[i][key_count] = np.where(values == sorted_spiketrains[i].annotations[group_key])[0][0]
                groupsizes[-1] += 1
                i += 1
            groupsizes.append(0)
            ref = i
        group_key_nbr = np.where(key_list == sorted_spiketrains[i].annotations[group_key])
    # group_key is now indicator for coloring
    # attribute array states for each key in key_list the unique value in form
    # of a numerical id.
    return attribute_array, group_key_nbr

def rasterplot(ax, spiketrain_list, key_list=[],
               markersize=4, markertype='.', bins=100, histscale=.1,
               style='ticks', palette='Set2'):
    """"""
    sns.set(style=style, palette=palette)
    sns.despine()
    sf = 1 - histscale
    left, bottom, width, height = ax.get_position()._get_bounds()
    ax.set_position([left, bottom, sf * width, sf * height])
    axhistx = plt.axes([left,               bottom + sf * height,
                        sf * width,         histscale * height])
    axhisty = plt.axes([left + sf * width,  bottom,
                        histscale * width,  sf * height])

    if isinstance(spiketrain_list[0], list):
        # var spiketrains is list of lists of spiketrains
        if isinstance(spiketrain_list[0][0], list):
            raise NotImplementedError('List of lists of lists of spiketrains'
                                      'are not yet accepted.')
    else:
        # var spiketrains is list of spiketrains
        spiketrain_list = [spiketrain_list]
        # var spiketrains is now handled as list of list of spiketrains

    t_lims = [[(st.t_start, st.t_stop) for st in spiketrains]
              for spiketrains in spiketrain_list
              ]
    tmin = min([min(t_it, key=lambda f: f[0])[0] for t_it in t_lims])
    tmax = max([max(t_it, key=lambda f: f[1])[1] for t_it in t_lims])
    yids = np.arange(sum([stlist.__len__() for stlist in spiketrain_list]))

    for list_count, spiketrains in enumerate(spiketrain_list):
        nbr_of_drawn_sts = sum([stlist.__len__()
                                for stlist in spiketrain_list[:list_count]])
        spiketrains = sorted(spiketrains, key=lambda x: [x.annotations[key]
                                                         for key in key_list])
        attributes = get_attributes(spiketrains, key_list)
        # ToDo: implement colormap for plots which represents the information in attributes
        # ToDo: Turn single Key to  1 element keylist
        print attributes
        for st_count, st in enumerate(spiketrains):
            # sns.stripplot(st.times.magnitude,
            #               [st.annotations['id']] * st.__len__(),
            #               orient='h')
            ax.plot(st.times.magnitude,
                    [st.annotations['id'] + nbr_of_drawn_sts]*st.__len__(),
                    markertype, ms=markersize)

        # sns.plt.show()
        # y = np.concatenate(np.array(
        #     [[st.annotations['id']] * st.__len__() for st in spiketrains]))
        # x = np.concatenate(
        #     np.array([st.tmes.magnitude for st in spiketrains]))
        # g = sns.JointGrid(x,y)
        # g.plot(sns.stripplot,sns.distplot)

        axhisty.barh(np.array([st.annotations['id'] + nbr_of_drawn_sts
                               for st in spiketrains]),
                     np.array([st.times.__len__()
                               for st in spiketrains]))

        axhistx.hist(np.concatenate(spiketrains), bins)
        period = tmax - tmin
        ax.set_xlim(tmin - .01 * period, tmax + .01 * period)
        # id_span = maxid - minid
        # ax.set_ylim(minid - .01 * id_span, maxid + .01 * id_span)
        axhistx.set_xlim(ax.get_xlim())
        axhisty.set_ylim(ax.get_ylim())
        ax.set_xlabel('t [{0}]'.format(spiketrains[0].units.dimensionality))
        ax.set_ylabel('Unit ID')
        axhistx.get_xaxis().set_visible(False)
        axhistx.get_yaxis().set_visible(False)
        axhisty.get_xaxis().set_visible(False)
        axhisty.get_yaxis().set_visible(False)

    plt.draw()

    return ax, axhistx, axhisty
