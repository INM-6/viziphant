"""
ToDo: Annotation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_attributes(spiketrains, key_list):
    """Spiketrains must be sorted according to keylist"""
    # find minimal grouping key:
    groupsizes = [0]
    key_count = len(key_list)
    attribute_array = np.zeros((spiketrains.__len__(), len(key_list)))
    maxgroupsizes = np.array([])
    group_bound = np.inf
    # count all group sizes for all keys in keylist:
    while key_count > 0 and max(groupsizes) < group_bound:
        key_count -= 1
        group_key = key_list[key_count]
        groupsizes = [0]
        i = 0
        ref = 0
        values = np.array([])
        # count all group sizes for values of current key:
        while i < spiketrains.__len__():
            if not np.where(values == spiketrains[i].annotations[group_key])[0]:
                # ToDo: improve if statement
                values = np.append(values, spiketrains[i].annotations[group_key])
            # count group size for a valuee of the current key:
            while i < spiketrains.__len__() and \
                    (spiketrains[i].annotations[group_key]
                         == spiketrains[ref].annotations[group_key]):
                attribute_array[i][key_count] = np.where(values == spiketrains[i].annotations[group_key])[0]
                groupsizes[-1] += 1
                i += 1
            groupsizes.append(0)
            ref = i
        maxgroupsizes = np.append(maxgroupsizes, max(groupsizes))
    if np.where(maxgroupsizes >= 2)[0].size:
        mingroupkey = max(np.where(maxgroupsizes >= 2)[0])
    else:
        mingroupkey = 0
    # mingroupkey is default indicator for coloring
    # attribute array states for each key in key_list the unique value in form of a numerical id.
    return attribute_array, mingroupkey


def rasterplot(ax, spiketrain_list, key_list=[], groupingdepth=1, spacing=6,
               markersize=4, markertype='.', bins=100, histscale=.1,
               style='ticks', palette='Set2'):
    # ToDo: nan oder '' in keylist fuer list position in sorting
    # ToDo: grouping seperator (linestyle + spacing)
    # ToDo: coloring key ('list', keyX)
    # ToDo: annotation->color dict
    # ToDo: PTSHs according to colors
    # ToDo: include/exclude dicts
    """"""
    sns.set(style=style, palette=palette)
    sns.despine()
    margin = 1 - histscale
    left, bottom, width, height = ax.get_position()._get_bounds()
    ax.set_position([left, bottom, margin * width, margin * height])
    axhistx = plt.axes([left, bottom + margin * height,
                        margin * width, histscale * height])
    axhisty = plt.axes([left + margin * width, bottom,
                        histscale * width, margin * height])

    if isinstance(spiketrain_list[0], list):
        # var spiketrains is list of lists of spiketrains
        placeholder = 0
    else:
        # var spiketrains is list of spiketrains
        spiketrain_list = [spiketrain_list]
        # var spiketrains is now handled as list of list of spiketrains

    if type(key_list) == 'str':
        key_list = [key_list]
    if '' not in key_list:
        key_list = [''] + key_list
    # ToDO:
    # create attribute tensor before going into the loop
    # flatten spiketrain list and add 'list' values to the attribute tensor
    # if necessary resort the spiketrainlist for the new attribute order
    # loop through the first $groupingdepth key

    assert groupingdepth >= 2, "Grouping in limited to two layers"


    t_lims = [[(st.t_start, st.t_stop) for st in spiketrains]
              for spiketrains in spiketrain_list
             ]
    tmin = min([min(t_it, key=lambda f: f[0])[0] for t_it in t_lims])
    tmax = max([max(t_it, key=lambda f: f[1])[1] for t_it in t_lims])
    period = tmax - tmin
    yticks = np.zeros(sum([stlist.__len__() for stlist in spiketrain_list]))

    for list_count, spiketrains in enumerate(spiketrain_list):
        nbr_of_drawn_sts = sum([stlist.__len__()
                                for stlist in spiketrain_list[:list_count]])
        spiketrains = sorted(spiketrains,
                             key=lambda x: [x.annotations[key]
                                            for key in key_list])
        attribute_array, mingroupkey = get_attributes(spiketrains, key_list)
        value_array = attribute_array[:, 0]

        # Dot display
        for st_count, st in enumerate(spiketrains):
            value_nbr = value_array[st_count]
            ax.plot(st.times.magnitude,
                    [st_count + nbr_of_drawn_sts  # account for prev lists
                     + groupingdepth*list_count*spacing  # list grouping
                     + groupingdepth/2*value_nbr*spacing  # key grouping
                     ] * st.__len__(),
                    markertype, ms=markersize)

        # Firing Rate histogram (right side)
        values, values_indices, groupsizes = np.unique(value_array,
                                                       return_counts=True,
                                                       return_index=True)
        for value_nbr, idx, groupsize in zip(values,values_indices,groupsizes):
            ypos = nbr_of_drawn_sts + np.sum(groupsizes[:int(value_nbr)])
            ycoords = np.arange(int(groupsize)) \
                      + ypos \
                      + groupingdepth * list_count * spacing \
                      + groupingdepth / 2 * value_nbr * spacing

            firing_rates = [st.times.__len__() / period
                            for st in spiketrains[idx:idx+groupsize]]
            axhisty.barh(ycoords, np.array(firing_rates))

            yticks[ypos:ypos+groupsize] = ycoords

        # PSTH (upper side)
        axhistx.hist(np.concatenate(spiketrains), bins)

    ax.set_yticks(yticks)
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

    return ax, axhistx, axhisty
