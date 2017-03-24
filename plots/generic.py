"""
ToDo:
-   Alpha Testing
-   Write more Annotation
-   Beautify: Fonts, Fontsizes, borders, grid, background, ...
-   Improve axis handling (of seaborn)
-   (use elphant pophist function) -> is overly complicated
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from elephant.statistics import mean_firing_rate
from math import log10, floor


def round_to_1(x):
    rounded = round(x, -int(floor(log10(abs(x)))))
    return rounded, rounded > x


def _get_attributes(spiketrains, key_list):
    """
    attribute_array is of shape (len(spiketrains), len(key_list))
    and consists of numerical ids for each value of each key for each
    spiketrain.

    Passed spiketrains must be sorted according to keylist.
    Keylist must not be empty"""

    key_count = len(key_list)
    attribute_array = np.zeros((len(spiketrains), len(key_list)))
    # count all group sizes for all keys in keylist:
    while key_count > 0:
        key_count -= 1
        group_key = key_list[key_count]
        i = 0
        if group_key in spiketrains[i].annotations:
            current_value = spiketrains[i].annotations[group_key]
        else:
            # use placeholder value when key is not in annotations
            # of the current spiketrain
            current_value = '####BLANK####'
        ref_value = current_value
        values = np.array([])
        # count all group sizes for values of current key:
        while i < spiketrains.__len__():
            if not len(values) or current_value not in values:
                values = np.append(values, current_value)
            # count group size for a value of the current key:
            while i < len(spiketrains) and current_value == ref_value:
                attribute_array[i][key_count] = \
                np.where(values == current_value)[0][0]
                i += 1
                if i < len(spiketrains):
                    if group_key in spiketrains[i].annotations:
                        current_value = spiketrains[i].annotations[
                            group_key]
                    else:
                        current_value = '####BLANK####'
            ref_value = current_value
    return attribute_array


def rasterplot(spiketrain_list,
               key_list=[],
               groupingdepth=0,
               spacing=[5,3],
               colorkey=0,
               pophist_mode='color',
               pophistbins=100,
               right_histogram=mean_firing_rate,
               filter_function=None,
               histscale=.1,
               labelkey=None,
               markerargs={'markersize':4,'marker':'.'},
               seperatorargs={'linewidth':1, 'linestyle':'--', 'color':'grey'},
               legend=False,
               legendargs={'loc':(1.,1.), 'markerscale':1.5, 'handletextpad':0},
               ax=None,
               style='ticks',
               palette='Set2',
               context='talk',
               colorcodes='colorblind'):

    """
    ----Write sth about function----

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
        and optionally by a line specefied by the the seperator parameter.
    :param spacing: int | [int, int]
        Size of whitespace seperating the groups. When groupingdepth = 2
        a list of two values can specify the distance between the groups in
        level 1 and level 2. When only only one value is given, the first level
        distance is 2 x spacing.
    :param colorkey: str | int  (default '')
        Contrasts values of a key by color. The key can be defined by its
        namestring or its position in key_list. Note that position 0 points to
        the list seperation key ('') which has default position 0 if not
        otherwise specified in key_list.
    :param pophist_mode: 'color' (default) | 'total'
         * total: One population histogram for all drawn spiketrains
         * color: Additionally to the total population histogram,
                  a histogram for each colored subset is drawn (see colorkey).
    :param pophistbins: int (default 100)
        Number of bins of the population histogram.
    :param right_histogram: function
        The function gets one neo.SpikeTrain as argument and has to return a
        scalar. For example the function in the elephant.statistics module can
        be used. (default: mean_firing_rate)
        When a function is applied is is recommende to set the axis visible
        and add an adequate axis label:
        axhisty.get_xaxis().set_visible(True)
        axhisty.set_xlabel('<FR>')
    :param filter_function: function
        The function gets a neo.SpikeTrain as argument and if the return is
        Truethy the spiketrain is included and if the return is Falsely it is
        exluded.
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
        Note that only groups (-> grouping) can be labeled as bulks.
        Alternatively you can color for an annotation key and show a legend.
    :param markerargs: dict
        Arguments dictionary is passed on to matplotlib.pyplot.plot()
    :param seperatorargs: dict | [dict, dict]
        If dict the arguments are applied to both types of seperators and if
        list of dicts the arguments can be specified for level 1 and 2
        seperators.
        Arguments dictionary is passed on to matplotlib.pyplot.plot()
    :param legend: boolean
    :param legendargs: dict
            Arguments dictionary is passed on to matplotlib.pyplot.legend()
    :param style:
    :param palette: string | sequence
        Define the color palette either by its name or use a custom palette in
        the form ([r,g,b],[r,g,b],...).
    :param colorcodes
    :param context
    :return: ax, axhistx, axhisty
    """
    # Initialize plotting canvas
    sns.set(style=style, palette=palette, context=context)

    if ax is None:
        fig = plt.figure()
        # axis must be created after sns.set() command for style to apply!
        ax = fig.add_subplot(111)

    sns.set_color_codes(colorcodes)

    margin = 1 - histscale
    left, bottom, width, height = ax.get_position()._get_bounds()
    ax.set_position([    left,                  bottom,
                         margin * width,        margin * height])
    axhistx = plt.axes([left,                   bottom + margin * height,
                        margin * width,         histscale * height])
    axhisty = plt.axes([left + margin * width,  bottom,
                        histscale * width,      margin * height])

    sns.despine(ax=axhistx)
    sns.despine(ax=axhisty)

    # Whitespace margin around dot display = 2%
    ws_margin = 0.02

    # Assertions
    list_key = "%$\@[#*&/!"  # will surely be unique ;)

    assert groupingdepth <= 2, "Grouping is limited to two layers"
    groupingdepth = int(groupingdepth)

    if type(key_list) == 'str':
        key_list = [key_list]

    if type(spacing) == list:
        if len(spacing) == 1:
            spacing = [spacing[0], spacing[0]]
    else:
        spacing = [spacing, spacing/2.]
    assert spacing[0] >= spacing[1]

    if type(colorkey) == int and len(key_list):
        assert colorkey < len(key_list)
        colorkey = key_list[colorkey]
    else:
        if not colorkey:
            colorkey = list_key
        else:
            assert colorkey in key_list

    if legend:
        assert len(key_list) > 0

    if labelkey == '':
        labelkey = list_key

    if type(seperatorargs) == list:
        assert len(seperatorargs) == 2
        assert type(seperatorargs[0]) == dict
        assert type(seperatorargs[1]) == dict
    else:
        if 'c' in seperatorargs:
            seperatorargs['color'] = seperatorargs['c']
        elif 'color' not in seperatorargs:
            seperatorargs['color'] = '#DDDDDD'
        seperatorargs = [seperatorargs, seperatorargs]

    markerargs['linestyle'] = ''

    # Flatten list of lists while keeping the grouping info in annotations
    if isinstance(spiketrain_list[0], list) and len(spiketrain_list) > 1:
        if '' not in key_list:
            key_list = [list_key] + key_list
        else:
            key_list = [list_key if not key else key for key in key_list]

        for list_nbr, st_list in enumerate(spiketrain_list):
            for st in st_list:
                st.annotations[list_key] = "set {}".format(list_nbr)
        spiketrain_list = [st for sublist in spiketrain_list for st in sublist]
    elif isinstance(spiketrain_list[0], list) and len(spiketrain_list) == 1:
        spiketrain_list = spiketrain_list[0]

    # Assertions on flattened list
    assert len(key_list) >= groupingdepth

    # Filter spiketrains according to given filter function
    if filter_function is not None:
        filter_index = []
        for st_count, spiketrain in enumerate(spiketrain_list):
            if filter_function(spiketrain):
                filter_index += [st_count]
        spiketrain_list = [spiketrain_list[i] for i in filter_index]

    # Initialize plotting parameters
    t_lims = [(st.t_start, st.t_stop) for st in spiketrain_list]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]
    period = tmax - tmin
    ax.set_xlim(tmin - ws_margin*period, tmax + ws_margin*period)
    yticks = np.zeros(len(spiketrain_list))

    # Sort spiketrains according to keylist
    sort_func = lambda x: ['' if key not in x.annotations
                           else x.annotations[key] for key in key_list]
    spiketrain_list = sorted(spiketrain_list, key=lambda x: sort_func(x))
    if len(key_list) > 1:
        attribute_array = _get_attributes(spiketrain_list, key_list)
    elif len(key_list) == 1:
        attribute_array = np.zeros((len(spiketrain_list), 2))
        attribute_array[:,0] = _get_attributes(spiketrain_list, key_list)[:,0]
    else:
        attribute_array = np.zeros((len(spiketrain_list), 1))

    # Define colormap
    if not len(key_list):
        nbr_of_colors = 1
        colorkey = None
    else:
        colorkey = np.where(colorkey == np.array(key_list))[0][0]
        nbr_of_colors = int(max(attribute_array[:, colorkey]) + 1)

    colormap = sns.color_palette(palette, nbr_of_colors)

    # Draw Population Histogram (upper side)
    sum_color = seperatorargs[0]['color']

    if pophist_mode == 'color' and colorkey:
        colorkeyvalues = np.unique(attribute_array[:, colorkey])
        if len(sns.color_palette()) < len(colorkeyvalues):
            print "\033[31mWarning: There are more subsets than can be " \
                  "seperated by colors in the color palette which might lead "\
                  "to confusion!\033[0m"
        for value in colorkeyvalues:
            idx = np.where(attribute_array[:, colorkey] == value)[0]
            axhistx.hist([stime for strain in [spiketrain_list[i] for i in idx]
                          for stime in strain],
                         pophistbins, histtype='step', linewidth=1,
                         color=colormap[int(value)])

    histout = axhistx.hist([stime for strain in spiketrain_list
                                  for stime in strain],
                           pophistbins, histtype='step', linewidth=1,
                           color=sum_color)

    axhistx_ydim, up = round_to_1(np.max(histout[0]))
    if up:
        axhistx.set_ylim(0, axhistx_ydim)
    axhistx.set_yticks([axhistx_ydim])
    axhistx.set_yticklabels(['{:.0f}'.format(axhistx_ydim)])

    # Legend for colorkey
    if legend:
        __, index = np.unique(attribute_array[:, colorkey], return_index=True)
        legend_labels = [spiketrain_list[i].annotations[key_list[colorkey]]
                         for i in index]
        legend_handles = [0] * len(index)

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
    # [ [ []..[] ] .... [ []..[] ] ] spiketrain_list
    # [ []..[] ] LIST
    # [] list
    # spiketrain

    # Loop through lists of lists of spiketrains
    for COUNT, SLIST in enumerate(spiketrain_list):

        # Seperator depth 1
        if COUNT:
            linepos = ypos + len(spiketrain_list[COUNT-1][-1]) \
                      + spacing[0]/2. - 0.5
            ax.plot(ax.get_xlim(), [linepos] * 2, **seperatorargs[0])

        # Loop through lists of spiketrains
        for count, slist in enumerate(SLIST):
            nbr_of_drawn_sts = int(sum([len(sl) for SL in
                                        spiketrain_list[:COUNT] for sl in SL])\
                                 + sum([len(sl) for sl in SLIST[:count]]))

            # Calculate postition of next spiketrain to draw
            prev_spaces = np.sum([len(SLIST_it) - 1
                                  for SLIST_it in spiketrain_list[:COUNT]])
            ypos = nbr_of_drawn_sts \
                 + int(bool(groupingdepth)) * COUNT * spacing[0] \
                 + groupingdepth/2 * count * spacing[1] \
                 + groupingdepth/2 * prev_spaces * spacing[1]

            # Separator depth 2
            if count:
                linepos = ypos - (spacing[1] + 1) / 2.
                ax.plot(ax.get_xlim(), [linepos] * 2, **seperatorargs[1])

            # Loop through spiketrains
            for st_count, st in enumerate(slist):
                current_st = nbr_of_drawn_sts + st_count
                annotation_value = int(attribute_array[current_st, colorkey])
                color = colormap[annotation_value]

                # Dot display
                handle = ax.plot(st.times.magnitude,
                                 [st_count + ypos] * st.__len__(), color=color,
                                 **markerargs)
                if legend:
                    legend_handles[annotation_value] = handle[0]

                # Right side histogram bar
                barvalue = right_histogram(st)
                barwidth = .8
                axhisty.barh(bottom=st_count + ypos,
                             width=barvalue,
                             height=barwidth,
                             color=color,
                             edgecolor=color)

            # Append positions of spiketrains to tick list
            ycoords = np.arange(len(slist)) + ypos
            yticks[nbr_of_drawn_sts:nbr_of_drawn_sts+len(slist)] = ycoords

    # Plotting axis
    yrange = yticks[-1] - yticks[0]
    ax.set_ylim(int(yticks[0] - ws_margin*yrange),
                int(yticks[-1] + ws_margin*yrange))
    axhistx.set_xlim(ax.get_xlim())
    axhisty.set_ylim(ax.get_ylim())
    ax.set_xlabel('t [{}]'.format(spiketrain_list[0][0][0].units.dimensionality))
    axhistx.get_xaxis().set_visible(False)
    axhisty.get_yaxis().set_visible(False)

    # Set ticks and labels for right side histogram
    axhisty_xdim, up = round_to_1(axhisty.get_xlim()[-1])
    if up:
        axhistx.set_ylim(0, axhistx_ydim)
    axhisty.set_xticks([axhisty_xdim])
    axhisty.set_xticklabels(['{}'.format(axhisty_xdim)])

    # Y labeling
    if labelkey is not None:
        if key_list and labelkey == key_list[0]:
            if groupingdepth > 0:
                labelkey = 0
        elif len(key_list) > 1 and labelkey == key_list[1]:
            if groupingdepth > 1:
                labelkey = 1

        if type(labelkey) == int or labelkey == '0+1':
            labelpos = [[] for label_level in range(2)]
            labelname = [[] for label_level in range(2)]

            # Labeling depth 1 + 2
            values1, index1, counts1 = np.unique(attribute_array[:, 0],
                                                 return_index=True,
                                                 return_counts=True)

            for v1, i1, c1 in zip(values1, index1, counts1):
                st = spiketrain_list[int(v1)][0][0]
                if key_list[0] in st.annotations:
                    labelname[0] += [st.annotations[key_list[0]]]
                    if labelkey == '0+1':
                        labelname[0][-1] += ' '*5
                else:
                    labelname[0] += ['']

                labelpos[0] += [(yticks[i1] + yticks[i1+c1-1])/2.]

                # Labeling depth 2
                values2, index2, counts2 = np.unique(attribute_array[i1:i1+c1, 1],
                                                     return_index=True,
                                                     return_counts=True)

                if groupingdepth / 2 and labelkey:
                    for v2, i2, c2 in zip(values2, index2, counts2):
                        st = spiketrain_list[int(v1)][int(v2)][0]
                        if key_list[1] in st.annotations:
                            labelname[1] += [st.annotations[key_list[1]]]
                        else:
                            labelname[1] += ['']

                        labelpos[1] += [(yticks[i1+i2] + yticks[i1+i2+c2-1])/2.]

            # Set labels according to labelkey
            if type(labelkey) == int:
                ax.set_yticks(labelpos[1] if labelkey else labelpos[0])
                ax.set_yticklabels(labelname[1] if labelkey else labelname[0])

            elif labelkey == "0+1":
                ax.set_yticks(labelpos[0] + labelpos[1])
                ax.set_yticklabels(labelname[0] + labelname[1])

        else:
            # Annotatation key as labelkey
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
    else:
        ax.set_yticks([])
        ax.set_yticklabels([''])

    # Draw legend
    if legend:
        ax.legend(legend_handles, legend_labels, **legendargs)

    # Remove list_key from annotations
    for SLIST in spiketrain_list:
        for slist in SLIST:
            for st in slist:
                st.annotations.pop(list_key, None)

    return ax, axhistx, axhisty
