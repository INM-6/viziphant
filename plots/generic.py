"""
ToDo: Annotation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_attributes(spiketrains, key_list):
    """Spiketrains must be sorted according to keylist.
    Keylist must not be empty"""
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
            if not len(values) or spiketrains[i].annotations[group_key] not in values:
                values = np.append(values, spiketrains[i].annotations[group_key])
            # count group size for a value of the current key:
            while i < spiketrains.__len__() and (spiketrains[i].annotations[group_key]
                         == spiketrains[ref].annotations[group_key]):
                attribute_array[i][key_count] = \
                np.where(values == spiketrains[i].annotations[group_key])[0][0]
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
    return attribute_array


def rasterplot(spiketrain_list, key_list=[], groupingdepth=1, spacing=3,
               colorkey='', PSTH_mode='color', markersize=4, markertype='.',
               seperator='', bins=100, histscale=.1, labelkey=None, ax=plt.gca(),
               style='ticks', palette='Set2'):

    # ToDo: cope with nonexisting annotation for keys in get_attributes
    # ToDo: seperator dict for passing line arguments
    # ToDo: marker dict for passing marker arguments
    # ToDo: if labelkey = colorkey use coler for labels
    # ToDo: possibility to give optional seperator_args as dict
    # ToDo: include/exclude dicts with custom selection statement
    # ToDo: elphant PSTH
    # ToDo: optional legend for color
    # ToDo: right-side hist mit custom function (i.e elephants)

    """
    :param ax: matplotlib axis
        If undefined takes the current axis.
    :param spiketrain_list: list
        list can either contain neo spiketrains or lists of spiketrains.
    :param key_list: string | list of strings
        Annotation key for which the spiketrains should be ordered. When list
        of keys is given spiketrains are ordered successively for the keys.roup
        By default the ordering by the given list of (lists of) spiketrains
        have priority. This can be bypassed by using an empty string ''
        as list-key at any position in the key_list.
    :param groupingdepth: 0 | 1 | 2
        * 0: No grouping
        * 1: grouping by given lists when list of list of spiketrains are given
             else grouping by first key
        * 2: additional grouping by first key or second key respectively
        The groups are sperated by whitespace specfied in the spacing parameter
        and optionally by a line specefied by the teh seperator parameter.
    :param spacing: int | [int, int]
        Size of whitespace seperating the groups. When groupingdepth = 2
        a list of two values can specify the distance between the groups in
        level 1 and level 2. When only only one value is given, the first level
        distance is 2 x spacing.
    :param colorkey: str | int  (default 0)
        Contrasts values of a key by color. The key can be defined by its
        namestring or its position in key_list. Note that position 0 points to
        the list seperation key ('') which has default position 0 if not
        otherwise specified in key_list.
    :param PSTH_mode: 'color' (default) | 'total'
         * color: For subset of the colorkey argument a seperate overlapping
                  PSTH is drawn.
         * totla: One PSTH for all drawn spiketrains
    :param markersize:
    :param markertype:
    :param seperator:
    :param bins: int (default 100)
        Number of bins of the PSTH
    :param histscale: float (default .1)
        Portion of the figure used for the histograms on the right and upper
        side.
    :param labelkey: 0 , 1 , '0+1' (default), 'annotation key', None
        * 0: Label lists when lists of list of lists of spiketrains are given
             else label values for first key
        * 1: Label for first key or second key accordingly
        * '0+1': Two level labeling of 0 and 1
        * annotation-key: Labeling each spiketrain with its value for given key
        * None: No labeling
    :param style:
    :param palette: string | sequence
        Define the color palette either by its name or use a custom palette in
        the form ([r,g,b],[r,g,b],...).
    :return: ax, axhistx, axhisty
    """
    # Initialize plotting canvas
    sns.set(style=style, palette=palette)
    sns.despine()
    margin = 1 - histscale
    left, bottom, width, height = ax.get_position()._get_bounds()
    ax.set_position([left, bottom, margin * width, margin * height])
    axhistx = plt.axes([left, bottom + margin * height,
                        margin * width, histscale * height])
    axhisty = plt.axes([left + margin * width, bottom,
                        histscale * width, margin * height])

    # Assertions
    list_key = "%$\@[#*&/!"  # will surely be unique ;)

    assert groupingdepth <= 2, "Grouping is limited to two layers"

    if type(key_list) == 'str':
        key_list = [key_list]

    if type(spacing) == int:
        spacing = [spacing * 2, spacing]

    if not type(seperator) == list:
        seperator = [seperator, seperator]

    if type(colorkey) == int:
        assert colorkey < len(key_list)
        colorkey = key_list[colorkey]

    if labelkey == '':
        labelkey = list_key

    # Flatten list of lists while keeping the grouping info in annotations
    if isinstance(spiketrain_list[0], list):
        if '' not in key_list:
            key_list = [list_key] + key_list
        else:
            key_list = [list_key if not key else key for key in key_list]

        for list_nbr, st_list in enumerate(spiketrain_list):
            for st in st_list:
                st.annotations[list_key] = "list {}".format(list_nbr)
        spiketrain_list = [st for sublist in spiketrain_list for st in sublist]

    assert len(key_list) >= groupingdepth

    # Initialize plotting parameters
    t_lims = [(st.t_start, st.t_stop) for st in spiketrain_list]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]
    period = tmax - tmin
    ax.set_xlim(tmin - .01 * period, tmax + .01 * period)
    yticks = np.zeros(len(spiketrain_list))

    # Sort list according keylist
    sort_func = lambda x: ['' if key not in x.annotations else x.annotations[key] for key in key_list]
    spiketrain_list = sorted(spiketrain_list, key=lambda x: sort_func(x))
    if key_list:
        # if len(key_list) == 1:
        #     attribute_array = np.zeros((len(spiketrain_list), 2))
        #     attribute_array[:, 0] = get_attributes(spiketrain_list, key_list)
        # else:
        attribute_array = get_attributes(spiketrain_list, key_list)
    else:
        attribute_array = np.zeros((len(spiketrain_list), 1))

    # Define colormap
    if not colorkey:
        colorkey = list_key
    colorkey = np.where(colorkey == np.array(key_list))[0]
    if not len(key_list):
        nbr_of_colors = 1
        colorkey = 0
    else:
        nbr_of_colors = int(max(attribute_array[:, colorkey])+1)
    colormap = sns.color_palette(palette, nbr_of_colors)

    # Draw PSTH (upper side)
    if PSTH_mode == 'color':
        # Check if colors are repeated and give depreciation warning
        for value in np.unique(attribute_array[:, colorkey]):
            idx = np.where(attribute_array[:, colorkey] == value)[0]
            axhistx.hist([stime for strain in [spiketrain_list[i] for i in idx]
                          for stime in strain], bins, color=colormap[int(value)])
    else:
        axhistx.hist([stime for strain in spiketrain_list for stime in strain],
                     bins,
                     color=sns.color_palette(palette, nbr_of_colors+1)[-1])

    # Reshape list into sublists according to groupingdepth
    if groupingdepth > 0:
        value1, index1, counts1 = np.unique(attribute_array[:, 0],
                                            return_index=True,
                                            return_counts=True)
        for v1, i1, c1 in zip(value1, index1, counts1):
            v1 = int(v1)
            spiketrain_list[v1:v1 + c1] = [spiketrain_list[v1:v1 + c1]]
            if groupingdepth > 1:
                value2, counts2 = np.unique(attribute_array[i1:i1 + c1, 1],
                                            return_counts=True)
                for v2, c2 in zip(value2, counts2):
                    v2 = int(v2)
                    spiketrain_list[v1][v2:v2+c2] = [spiketrain_list[v1][v2:v2+c2]]
            else:
                spiketrain_list[v1] = [spiketrain_list[v1]]
    else:
        spiketrain_list = [[spiketrain_list]]

    # HIERARCHIE:
    # [[[]..[]]...[[]..[]]] spiketrain_list
    # [[]..[]] LIST
    # [] list
    # spiketrain

    # Loop through lists of lists of spiketrains
    for COUNT, SLIST in enumerate(spiketrain_list):
        for count, slist in enumerate(SLIST):
            nbr_of_drawn_sts = int(sum([len(sl) for SL in
                                        spiketrain_list[:COUNT] for sl in SL])\
                                 + sum([len(sl) for sl in SLIST[:count]]))

            ypos = nbr_of_drawn_sts \
                 + groupingdepth * COUNT * spacing[0] \
                 + groupingdepth / 2 * count * spacing[1]

            for st_count, st in enumerate(slist):
                color = colormap[int(attribute_array[nbr_of_drawn_sts,colorkey])]

                # Dot display
                ax.plot(st.times.magnitude,
                        [st_count + ypos] * st.__len__(),
                        markertype, ms=markersize, color=color)

                # Firing Rate histogram (right side)
                axhisty.barh(st_count + ypos - .5, st.times.__len__(),
                             height=1., color=color)

            ycoords = np.arange(len(slist)) + ypos
            yticks[nbr_of_drawn_sts:nbr_of_drawn_sts+len(slist)] = ycoords

            # Seperator depth 2
            if count < len(SLIST) - 1:
                linepos = ypos + len(slist) + (spacing[1]-1)/2.
                ax.plot(ax.get_xlim(), [linepos] * 2,
                        linestyle=seperator[1], linewidth=markersize/4.,
                        color='grey')

        # Seperator depth 1
        if COUNT < len(spiketrain_list) - 1:
            linepos = ypos + len(SLIST[-1]) \
                      + (spacing[0]-1*groupingdepth)/2.*groupingdepth
            ax.plot(ax.get_xlim(), [linepos] * 2,
                    linestyle=seperator[1], linewidth=markersize / 4.,
                    color='red')

    # Plotting axis
    axhistx.set_xlim(ax.get_xlim())
    axhisty.set_ylim(ax.get_ylim())
    ax.set_xlabel('t [{0}]'.format(spiketrain_list[0][0][0].units.dimensionality))
    ax.set_ylabel('')
    axhistx.get_xaxis().set_visible(False)
    axhistx.get_yaxis().set_visible(False)
    axhisty.get_xaxis().set_visible(False)
    axhisty.get_yaxis().set_visible(False)

    # Y labeling
    if labelkey is not None:
        if key_list and labelkey == key_list[0]:
            labelkey = 0
        elif len(key_list) > 1 and labelkey == key_list[1]:
            labelkey = 1

        if type(labelkey) == int or labelkey == '0+1':
            labelpos = [[] for label_level in range((groupingdepth / 2 + 1))]
            labelname = [[] for label_level in range((groupingdepth / 2 + 1))]

            values1, index1, counts1 = np.unique(attribute_array[:, 0],
                                                 return_index=True,
                                                 return_counts=True)

            for v1, i1, c1 in zip(values1, index1, counts1):
                st = spiketrain_list[int(v1)][0][0]
                if key_list[0] in st.annotations:
                    labelname[0] += [st.annotations[key_list[0]]]
                else:
                    labelname[0] += ['']
                labelpos[0] += [yticks[i1 + c1/2]]
                values2, index2, counts2 = np.unique(attribute_array[i1:i1+c1, 1],
                                                     return_index=True,
                                                     return_counts=True)

                if groupingdepth/2 and labelkey:
                    for v2, i2, c2 in zip(values2, index2, counts2):
                        st = spiketrain_list[int(v1)][int(v2)][0]
                        if key_list[1] in st.annotations:
                            labelname[1] += [st.annotations[key_list[1]]]
                        else:
                            labelname[1] += ['']
                        labelpos[1] += [yticks[i1 + i2 + c2 / 2]]


            if type(labelkey) == int:
                ax.set_yticks(labelpos[1] if labelkey else labelpos[0])
                ax.set_yticklabels(labelname[1] if labelkey else labelname[0])

            elif labelkey == "0+1":
                ax.set_yticks(labelpos[0] + labelpos[1])
                ax.set_yticklabels(labelname[0] + labelname[1])

        else:
            labelname = []
            for COUNT, SLIST in enumerate(spiketrain_list):
                for count, slist in enumerate(SLIST):
                    for st_count, st in enumerate(slist):
                        if labelkey in st.annotations:
                            labelname += [st.annotations[labelkey]]
                        else:
                            labelname += ['']
            ax.set_yticks(yticks)
            ax.set_yticklabels(labelname)


    # legend for colorkey#


    return ax, axhistx, axhisty
