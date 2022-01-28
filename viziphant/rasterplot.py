"""
Raster and event plots of spike times
-------------------------------------

.. autosummary::
    :toctree: toctree/rasterplot/

    eventplot
    rasterplot
    rasterplot_rates

"""
# Copyright 2017-2022 by the Viziphant team, see `doc/authors.rst`.
# License: Modified BSD, see LICENSE.txt for details.

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import seaborn as sns
import warnings
import neo
from math import log10, floor

from elephant.statistics import mean_firing_rate
from viziphant.utils import check_same_units


def _round_to_1(x):
    rounded = round(x, -int(floor(log10(abs(x)))))
    return rounded, rounded > x


def _get_attributes(spiketrains, key_list):
    """
    This function returns attribute_array which is of an array of shape
    (len(spiketrains), len(key_list)) and consists of numerical ids for each
    value of each key for each spike train.

    Passed spike trains must be already sorted according to key_list and
    key_list must not be empty.
    """

    key_count = len(key_list)
    attribute_array = np.zeros((len(spiketrains), len(key_list)))
    # count all group sizes for all keys in key_list:
    while key_count > 0:
        key_count -= 1
        group_key = key_list[key_count]
        i = 0
        if group_key in spiketrains[i].annotations:
            current_value = spiketrains[i].annotations[group_key]
        else:
            # use placeholder value when key is not in annotations
            # of the current spike train
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


def rasterplot_rates(spiketrains,
                     key_list=[],
                     groupingdepth=0,
                     spacing=[8, 3],
                     colorkey=0,
                     pophist_mode='color',
                     pophistbins=100,
                     right_histogram=mean_firing_rate,
                     righthist_barwidth=1.01,
                     filter_function=None,
                     histscale=.1,
                     labelkey=None,
                     markerargs={'markersize': 4, 'marker': '.'},
                     separatorargs=[
                         {'linewidth': 2, 'linestyle': '--', 'color': '0.8'},
                         {'linewidth': 1, 'linestyle': '--', 'color': '0.8'}],
                     legend=False,
                     legendargs={'loc': (.98, 1.), 'markerscale': 1.5,
                                 'handletextpad': 0},
                     ax=None,
                     style='ticks',
                     palette=None,
                     context=None,  # paper, poster, talk
                     ):
    """
    This function plots the dot display of spike trains alongside its
    population histogram and the mean firing rate (or a custom function).

    Optional visual aids are offered such as sorting, grouping and color coding
    on the basis of the arrangement in list of spike trains and spike train
    annotations.
    Changes to optics of the dot marker, the separators and the legend can be
    applied by providing a dict with the respective parameters. Changes and
    additions to the dot display itself or the two histograms are best realized
    by using the returned axis handles.


    Parameters
    ----------
    spiketrains: list of neo.SpikeTrain or list of list of neo.SpikeTrain
        List can either contain Neo SpikeTrains object or lists of Neo
        SpikeTrains objects.
    key_list: str or list of str
        Annotation key(s) for which the spike trains should be ordered.
        When list of keys is given the spike trains are ordered successively
        for the keys.
        By default the ordering by the given lists of spike trains have
        priority. This can be bypassed by using an empty string '' as list-key
        at any position in the key_list.
    groupingdepth: int
        * 0: No grouping (default)
        * 1: grouping by first key in key_list.
             Note that when list of lists of spike trains are given the first
             key is by the list identification key ''. If this is unwanted
             the empty string '' can be placed at a different position in
             key_list.
        * 2: additional grouping by second key respectively

        The groups are separated by whitespace specified in the spacing
        parameter and optionally by a line specified by the the separatorargs.
    spacing: int or list of int
        Size of whitespace separating the groups in units of spike trains.
        When groupingdepth == 2 a list of two values can specify the distance
        between the groups in level 1 and level 2. When only one value is given
        level 2 spacing is set to half the spacing of level 1.
        Default: [5, 3]
    colorkey: str or int  (default 0)
        Contrasts values of a key by color. The key can be defined by its
        namestring or its position in key_list. Note that position 0 points to
        the list identification key ('') when list of lists of spike trains are
        given, if not otherwise specified in key_list!
    pophist_mode: str
         * total: One population histogram for all drawn spike trains

         * color: Additionally to the total population histogram,
                  a histogram for each colored subset is drawn (see colorkey).
    pophistbins: int (default 100)
        Number of bins used for the population histogram.
    right_histogram: function
        The function gets ONE neo.SpikeTrain object as argument and has to
        return a scalar.
        For example the functions in the elephant.statistics module can
        be used. (default: mean_firing_rate)
        When a function is applied is is recommended to set the axis label
        accordingly by using the axis handle returned by the function:
        axhisty.set_xlabel('Label Name')
    righthist_barwidth: float (default 1.01)
        The bin width of the right side histogram.
    filter_function: function
        The function gets ONE neo.SpikeTrain object as argument and if the
        return is True the spike train is included; if False it is exluded.
    histscale: float (default .1)
        Portion of the figure used for the histograms on the right and upper
        side.
    labelkey: int or string or None
        * 0, 1: Set label according to first or second key in key_list.
                Note that the first key is by default the list identification
                key ('') when list of lists of spike trains are given.
        * '0+1': Two level labeling of 0 and 1
        * annotation-key: Labeling each spike train with its value for given
                          key
        * None: No labeling

        Note that only groups (-> see groupingdepth) can be labeled as bulks.
        Alternatively you can color for an annotation key and show a legend.
    markerargs: dict
        Arguments dictionary is passed on to matplotlib.pyplot.plot()
    separatorargs: dict or list of dict or None
        If only one dict is given and groupingdepth == 2 the arguments are
        applied to the separator of both level. Otherwise the arguments are
        of separatorargs[0] are applied to the level 1 and [1] to level 2.
        Arguments dictionary is passed on to matplotlib.pyplot.plot()
        To turn of separators set it to None.
    legend: bool
        Show legend?
    legendargs: dict
        Arguments dictionary is passed on to matplotlib.pyplot.legend()
    ax: matplotlib axis or None (default)
        The axis onto which to plot. If None a new figure is created.
        When an axis is given, the function can't handle the figure settings.
        Therefore it is recommended to call seaborn.set() with your preferred
        settings before creating your matplotlib figure in order to control
        your plotting layout.
    style: str
        seaborn style setting. Default: 'ticks'
    palette: str or sequence
        Define the color palette either by its name or use a custom palette in
        a sequence of the form ([r,g,b],[r,g,b],...).
    context: str
        'paper'(default) | 'talk' | 'poster'
        seaborn context setting which controls the scaling of labels. For the
        three options the parameters are scaled by .8, 1.3, and 1.6
        respectively.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The handle of the dot display plot.
    axhistx : matplotlib.axes.Axes
        The handle of the histogram plot above the the dot display
    axhisty : matplotlib.axes.Axes
        The handle of the histogram plot on the right hand side

    See Also
    --------
    rasterplot : simplified raster plot
    eventplot : plot spike times in vertical stripes

    Examples
    --------
    1. Basic Example.

    .. plot::
        :include-source:

        from elephant.spike_train_generation import \
            homogeneous_poisson_process, homogeneous_gamma_process
        import quantities as pq
        import matplotlib.pyplot as plt
        from viziphant.rasterplot import rasterplot_rates

        spiketrains = [homogeneous_poisson_process(rate=10 * pq.Hz)
                       for _ in range(100)]
        rasterplot_rates(spiketrains)
        plt.show()

    2. Plot visually separated realizations of different neurons.

    .. plot::
        :include-source:

        from elephant.spike_train_generation import \
            homogeneous_poisson_process, homogeneous_gamma_process
        import quantities as pq
        import matplotlib.pyplot as plt
        from viziphant.rasterplot import rasterplot_rates

        spiketrains1 = [homogeneous_poisson_process(rate=10 * pq.Hz)
                        for _ in range(100)]
        spiketrains2 = [homogeneous_gamma_process(a=3, b=10 * pq.Hz)
                        for _ in range(100)]
        rasterplot_rates([spiketrains1, spiketrains2])
        plt.show()

    3. Add annotations to spike trains.

    .. plot::
        :include-source:

        from elephant.spike_train_generation import \
            homogeneous_poisson_process, homogeneous_gamma_process
        import quantities as pq
        import matplotlib.pyplot as plt
        from viziphant.rasterplot import rasterplot_rates

        spiketrains1 = [homogeneous_poisson_process(rate=10 * pq.Hz)
                        for _ in range(100)]
        spiketrains2 = [homogeneous_gamma_process(a=3, b=10 * pq.Hz)
                        for _ in range(100)]
        for i, (st1, st2) in enumerate(zip(spiketrains1, spiketrains2)):
            if i % 2 == 1:
                st1.annotations['parity'] = 'odd'
                st2.annotations['parity'] = 'odd'
            else:
                st1.annotations['parity'] = 'even'
                st2.annotations['parity'] = 'even'

        # plot separates the lists and the annotation values within each list
        rasterplot_rates([spiketrains1, spiketrains2], key_list=['parity'],
                          groupingdepth=2, labelkey='0+1')

    ``''`` key change the priority of the list grouping:

    .. code-block:: python

        rasterplot_rates([spiketrains1, spiketrains2],
                          key_list=['parity', ''],
                          groupingdepth=2, labelkey='0+1')

    Groups can also be emphasized by an explicit color mode:

    .. code-block:: python

        rasterplot_rates([spiketrains1, spiketrains2],
                          key_list=['', 'parity'],
                          groupingdepth=1, labelkey=0, colorkey='parity',
                          legend=True)

    """

    # Initialize plotting canvas
    sns.set_style(style)

    if context is not None:
        sns.set_context(context)

    if palette is not None:
        sns.set_palette(palette)
    else:
        palette = sns.color_palette()

    if ax is None:
        fig, ax = plt.subplots()
        # axis must be created after sns.set() command for style to apply!

    margin = 1 - histscale
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom,
                     margin * width, margin * height])
    axhistx = plt.axes([left, bottom + margin * height,
                        margin * width, histscale * height])
    axhisty = plt.axes([left + margin * width, bottom,
                        histscale * width, margin * height])

    sns.despine(ax=axhistx)
    sns.despine(ax=axhisty)

    # Whitespace margin around dot display = 2%
    ws_margin = 0.02

    # Control of user entries
    if groupingdepth > 2:
        raise ValueError("Grouping is limited to two layers.")
    groupingdepth = int(groupingdepth)

    list_key = r"%$\@[#*&/!"  # unique key to be added to annotations to store
    # list ordering information.

    if type(key_list) == 'str':
        key_list = [key_list]

    if '' not in key_list:
        key_list = [list_key] + key_list
    else:
        key_list = [list_key if not key else key for key in key_list]

    if type(spacing) == list:
        if len(spacing) == 1:
            spacing = [spacing[0], spacing[0] / 2.]
    else:
        spacing = [spacing, spacing / 2.]
    if spacing[0] < spacing[1]:
        raise DeprecationWarning("For reasonable visual aid, spacing between "
                                 "top level group (spacing[0]) must be larger "
                                 "than for subgroups (spacing[1]).")

    if type(colorkey) == int and len(key_list):
        if colorkey >= len(key_list):
            raise IndexError("An integer colorkey must refer to a position in "
                             "key_list.")
        colorkey = key_list[colorkey]
    else:
        if not colorkey:
            colorkey = list_key
        elif colorkey not in key_list:
            raise AttributeError("colorkey must be in key_list.")

    if legend and not key_list:
        raise AttributeError("Legend requires a non empty key_list.")

    if labelkey == '':
        labelkey = list_key

    if type(separatorargs) == list:
        if len(separatorargs) == 1:
            separatorargs += separatorargs
        for args in separatorargs:
            if type(args) != dict:
                raise TypeError("The parameters must be given as dict.")
    else:
        separatorargs = [separatorargs, separatorargs]

    for i, args in enumerate(separatorargs):
        if 'c' in args:
            separatorargs[i]['color'] = args['c']
        elif 'color' not in args:
            separatorargs[i]['color'] = '0.8'

    markerargs['linestyle'] = ''

    # Flatten list of lists while keeping the grouping info in annotations
    if isinstance(spiketrains[0], list):
        for list_nbr, st_list in enumerate(spiketrains):
            for st in st_list:
                st.annotations[list_key] = "set {}".format(list_nbr)
        spiketrains = [st for sublist in spiketrains for st in sublist]
    else:
        for st in spiketrains:
            st.annotations[list_key] = "set {}".format(0)
        key_list.remove(list_key)
        key_list.append(list_key)

    # Input checks on flattened lists
    if len(key_list) < groupingdepth:
        raise ValueError("Can't group more as keys in key_list.")

    # Filter spike trains according to given filter function
    if filter_function is not None:
        filter_index = []
        for st_count, spiketrain in enumerate(spiketrains):
            if filter_function(spiketrain):
                filter_index += [st_count]
        spiketrains = [spiketrains[i] for i in filter_index]

    # Initialize plotting parameters
    t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]
    period = tmax - tmin
    ax.set_xlim(tmin - ws_margin * period, tmax + ws_margin * period)
    yticks = np.zeros(len(spiketrains))

    # Sort spike trains according to keylist
    def sort_func(x):
        return ['' if key not in x.annotations
                else x.annotations[key] for key in key_list]

    spiketrains = sorted(spiketrains, key=lambda x: sort_func(x))
    if len(key_list) > 1:
        attribute_array = _get_attributes(spiketrains, key_list)
    elif len(key_list) == 1:
        attribute_array = np.zeros((len(spiketrains), 2))
        attribute_array[:, 0] = _get_attributes(spiketrains, key_list)[:, 0]
    else:
        attribute_array = np.zeros((len(spiketrains), 1))

    # Define colormap
    if not len(key_list):
        nbr_of_colors = 1
        colorkey = None
    else:
        colorkey = np.where(colorkey == np.array(key_list))[0][0]
        nbr_of_colors = int(max(attribute_array[:, colorkey]) + 1)

    colormap = sns.color_palette(palette, nbr_of_colors)

    # Draw population histogram (upper side)
    colorkeyvalues = np.unique(attribute_array[:, colorkey])

    if pophist_mode == 'color' and len(colorkeyvalues) - 1:
        if len(sns.color_palette()) < len(colorkeyvalues):
            warnings.warn("There are more subsets than can be separated by "
                          "colors in the color palette which might lead to "
                          "confusion")
        max_y = 0
        for value in colorkeyvalues:
            idx = np.where(attribute_array[:, colorkey] == value)[0]
            histout = axhistx.hist(
                np.concatenate([spiketrains[i] for i in idx]),
                pophistbins, histtype='step', linewidth=1,
                color=colormap[int(value)])
            max_y = np.max([max_y, np.max(histout[0])])

    else:  # pophist_mode == 'total':
        if len(colorkeyvalues) - 1:
            sum_color = separatorargs[0]['color']
        else:
            sum_color = sns.color_palette()[0]
        histout = axhistx.hist(np.concatenate(spiketrains),
                               pophistbins, histtype='step', linewidth=1,
                               color=sum_color)
        max_y = np.max(histout[0])

    # Set ticks and labels for population histogram
    axhistx_ydim, up = _round_to_1(max_y)
    if max_y > axhistx.get_ylim()[-1]:
        axhistx.set_ylim(0, max_y)
    if up and axhistx_ydim > max_y:
        axhistx.set_ylim(0, axhistx_ydim)
    axhistx.set_yticks([axhistx_ydim])
    axhistx.set_yticklabels(['{:.0f}'.format(axhistx_ydim)])
    axhistx.set_ylabel('count')

    # Legend for colorkey
    if legend:
        __, index = np.unique(attribute_array[:, colorkey], return_index=True)
        legend_labels = [spiketrains[i].annotations[key_list[colorkey]]
                         for i in index]
        legend_handles = [0] * len(index)

    # Reshape list into sublists according to groupingdepth
    if groupingdepth > 0:
        value1, index1, counts1 = np.unique(attribute_array[:, 0],
                                            return_index=True,
                                            return_counts=True)
        for v1, i1, c1 in zip(value1, index1, counts1):
            v1 = int(v1)
            spiketrains[v1:v1 + c1] = [spiketrains[v1:v1 + c1]]
            if groupingdepth > 1:
                __, counts2 = np.unique(attribute_array[i1:i1 + c1, 1],
                                        return_counts=True)
                for v2, c2 in enumerate(counts2):
                    v2 = int(v2)
                    spiketrains[v1][v2:v2 + c2] = [
                        spiketrains[v1][v2:v2 + c2]]
            else:
                spiketrains[v1] = [spiketrains[v1]]
    else:
        spiketrains = [[spiketrains]]

    # HIERARCHIE:
    # [ [ []..[] ] .... [ []..[] ] ] spiketrains
    # [ []..[] ] LIST
    # [] list
    # spike train

    # Loop through lists of lists of spike trains
    for COUNT, SLIST in enumerate(spiketrains):

        # Separator depth 1
        if COUNT and separatorargs is not None:
            linepos = ypos + len(spiketrains[COUNT - 1][-1]) \
                      + spacing[0] / 2. - 0.5
            ax.plot(ax.get_xlim(), [linepos] * 2, **separatorargs[0])

        # Loop through lists of spike trains
        for count, slist in enumerate(SLIST):
            nbr_of_drawn_sts = int(
                sum(len(sl) for SL in spiketrains[:COUNT] for sl in SL) +
                sum(len(sl) for sl in SLIST[:count]))

            # Calculate postition of next spike train to draw
            prev_spaces = np.sum([len(SLIST_it) - 1
                                  for SLIST_it in spiketrains[:COUNT]])
            ypos = nbr_of_drawn_sts + int(
                bool(groupingdepth)) * COUNT * spacing[0] \
                + groupingdepth / 2 * count * spacing[1] \
                + groupingdepth / 2 * prev_spaces * spacing[1]

            # Separator depth 2
            if count and separatorargs is not None:
                linepos = ypos - (spacing[1] + 1) / 2.
                ax.plot(ax.get_xlim(), [linepos] * 2, **separatorargs[1])

            # Loop through spike trains
            for st_count, st in enumerate(slist):
                current_st = nbr_of_drawn_sts + st_count
                annotation_value = int(attribute_array[current_st, colorkey])
                color = colormap[annotation_value]

                # Dot display
                handle = ax.plot(st.times.magnitude,
                                 [st_count + ypos] * st.__len__(),
                                 color=color, **markerargs)
                if legend:
                    legend_handles[annotation_value] = handle[0]

                # Right side histogram bar
                barvalue = right_histogram(st)
                barwidth = righthist_barwidth
                axhisty.barh(y=st_count + ypos,  # - barwidth/2.,
                             width=barvalue,
                             height=barwidth,
                             color=color,
                             edgecolor=color)

            # Append positions of spike trains to tick list
            ycoords = np.arange(len(slist)) + ypos
            yticks[nbr_of_drawn_sts:nbr_of_drawn_sts + len(slist)] = ycoords

    # Plotting axis
    yrange = yticks[-1] - yticks[0]
    ax.set_ylim(yticks[0] - ws_margin * yrange,
                yticks[-1] + ws_margin * yrange)
    axhistx.set_xlim(ax.get_xlim())
    axhisty.set_ylim(ax.get_ylim())
    ax.set_xlabel(f'Time ({spiketrains[0][0][0].units.dimensionality})')
    axhistx.get_xaxis().set_visible(False)
    axhisty.get_yaxis().set_visible(False)

    # Set ticks and labels for right side histogram
    axhisty_xdim, up = _round_to_1(axhisty.get_xlim()[-1])
    if up:
        axhistx.set_ylim(0, axhistx_ydim)
    axhisty.set_xticks([axhisty_xdim])
    axhisty.set_xticklabels(['{}'.format(axhisty_xdim)])

    # Y labeling
    if key_list and labelkey in key_list + [0, 1, '0+1']:
        if labelkey == key_list[0]:
            if groupingdepth > 0:
                labelkey = 0
        elif len(key_list) > 1 and labelkey == key_list[1]:
            if groupingdepth > 1:
                labelkey = 1

        if type(labelkey) == int or labelkey == '0+1':
            labelpos = [[] for label_level in range(2)]
            labelname = [[] for label_level in range(2)]

            # Labeling depth 1 + 2
            if groupingdepth:
                values1, index1, counts1 = np.unique(attribute_array[:, 0],
                                                     return_index=True,
                                                     return_counts=True)

                for v1, i1, c1 in zip(values1, index1, counts1):
                    st = spiketrains[int(v1)][0][0]
                    if key_list[0] in st.annotations:
                        labelname[0] += [st.annotations[key_list[0]]]
                        if labelkey == '0+1':
                            labelname[0][-1] += ' ' * 5
                    else:
                        labelname[0] += ['']

                    labelpos[0] += [(yticks[i1] + yticks[i1 + c1 - 1]) / 2.]

                    # Labeling depth 2
                    if groupingdepth / 2 and labelkey and len(key_list) - 1:
                        __, index2, counts2 = np.unique(
                            attribute_array[i1:i1 + c1, 1],
                            return_index=True,
                            return_counts=True)

                        for v2, (i2, c2) in enumerate(zip(index2, counts2)):
                            st = spiketrains[int(v1)][int(v2)][0]
                            if key_list[1] in st.annotations:
                                labelname[1] += [st.annotations[key_list[1]]]
                            else:
                                labelname[1] += ['']

                            labelpos[1] += [(yticks[i1 + i2] + yticks[
                                i1 + i2 + c2 - 1]) / 2.]

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
            for COUNT, SLIST in enumerate(spiketrains):
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

    # Draw legend
    if legend:
        ax.legend(legend_handles, legend_labels, **legendargs)

    # Remove list_key from annotations
    for SLIST in spiketrains:
        for slist in SLIST:
            for st in slist:
                st.annotations.pop(list_key, None)

    return ax, axhistx, axhisty


def rasterplot(spiketrains, axes=None, histogram_bins=0, title=None,
               color=None, **kwargs):
    """
    Simple and fast raster plot of spike times.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or pq.Quantity
        A list of `neo.SpikeTrain` objects or quantity arrays with spike times.
    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None
    histogram_bins : int, optional
        Defines the number of histogram bins. If set to ``0``, no histogram
        is shown.
        Default: 0
    title : str or None, optional
        The axes title.
        Default: None
    color : str or list of str or None, optional
        Raster colors.
        Default: None
    **kwargs
        Additional parameters passed to matplotlib `scatter` function.

    Returns
    -------
    axes : matplotlib.Axes.axes

    See Also
    --------
    rasterplot_rates : advanced raster plot
    eventplot : plot spike times in vertical stripes

    Examples
    --------
    1. Basic example.

    .. plot::
        :include-source:

        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.rasterplot import rasterplot
        np.random.seed(7)
        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        rasterplot(spiketrains, s=3, c='black')
        plt.show()

    2. Raster plot with a histogram and events.

    .. plot::
        :include-source:

        import neo
        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.rasterplot import rasterplot
        from viziphant.events import add_event

        np.random.seed(7)
        spiketrains = [homogeneous_poisson_process(rate=r * pq.Hz,
                       t_stop=10 * pq.s) for r in range(1, 21)]
        event = neo.Event([0.5, 2.8] * pq.s, labels=['Trig ON', 'Trig OFF'])

        axes = rasterplot(spiketrains, histogram_bins=50, title='Title', s=0.5)
        add_event(axes, event=event)
        plt.show()

    """
    if isinstance(spiketrains[0], neo.SpikeTrain):
        spiketrains = [spiketrains]
    spiketrains = list(filter(len, spiketrains))
    check_same_units(spiketrains)
    units = spiketrains[0][0].units
    if color is None:
        color = kwargs.pop('c', None)
    if not isinstance(color, (list, tuple)):
        color = [color] * len(spiketrains)

    if axes is None:
        nrows = 2 if histogram_bins else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=1)

    count = 0
    histtype = 'bar' if len(spiketrains) == 1 else 'step'
    for sts_population, c in zip(spiketrains, color):
        sts_population = [st.magnitude for st in sts_population]
        axes = np.atleast_1d(axes)
        times_population = np.hstack(sts_population)
        ys = np.hstack([np.repeat(i + count, repeats=len(st))
                        for i, st in enumerate(sts_population)])
        axes[0].scatter(times_population, ys, c=c, **kwargs)
        if histogram_bins:
            axes[1].hist(times_population, bins=histogram_bins,
                         histtype=histtype, color=c)
        count += len(sts_population)
    axes[0].set_yticks([0, count - 1])
    axes[0].set_title(title)

    if histogram_bins:
        axes[1].set_ylabel("Spike count")
    axes[-1].set_xlabel(f"Time ({units.dimensionality})")

    if len(axes) == 1:
        axes = axes[0]

    return axes


def eventplot(spiketrains, axes=None, histogram_bins=0, title=None, **kwargs):
    """
    Spike times eventplot with an additional histogram.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or pq.Quantity
        A list of `neo.SpikeTrain` objects or quantity arrays with spike times.
    axes : matplotlib.axes.Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
        Default: None
    histogram_bins : int, optional
        Defines the number of histogram bins. If set to ``0``, no histogram
        is shown.
        Default: 0
    title : str or None, optional
        The axes title.
        Default: None
    **kwargs
        Additional parameters passed to matplotlib `eventplot` function.

    Returns
    -------
    axes : matplotlib.axes.Axes

    See Also
    --------
    rasterplot : simplified raster plot
    rasterplot_rates : advanced raster plot

    Examples
    --------
    Basic spike times eventplot.

    .. plot::
        :include-source:

        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.rasterplot import eventplot
        np.random.seed(12)
        spiketrains = [homogeneous_poisson_process(rate=10*pq.Hz,
                       t_stop=10*pq.s) for _ in range(10)]
        eventplot(spiketrains, linelengths=0.75, color='black')
        plt.show()

    To plot with a histogram, provide a value for ``histogram_bins``.
    To compare spike times between different neurons, create
    `matplotlib.axes.Axes` instance prior to calling the function.
    Additionally, you can add events to the plot with
    :func:`viziphant.events.add_event` function.

    .. plot::
        :include-source:

        import neo
        import numpy as np
        import quantities as pq
        import matplotlib.pyplot as plt
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.rasterplot import eventplot
        from viziphant.events import add_event
        np.random.seed(12)
        spiketrains = [homogeneous_poisson_process(rate=5*pq.Hz,
                       t_stop=10*pq.s) for _ in range(20)]

        fig, axes = plt.subplots(2, 2, sharex=True, sharey='row')
        event = neo.Event([0.5, 8]*pq.s, labels=['trig0', 'trig1'])
        eventplot(spiketrains[:10], axes=axes[:, 0], histogram_bins=20,
                  title="Neuron A", linelengths=0.75, linewidths=1)
        add_event(axes[:, 0], event)
        eventplot(spiketrains[10:], axes=axes[:, 1], histogram_bins=20,
                  title="Neuron B", linelengths=0.75, linewidths=1)
        add_event(axes[:, 1], event)
        plt.show()

    """
    check_same_units(spiketrains)
    units = spiketrains[0].units
    spiketrains = [st.magnitude for st in spiketrains]
    if axes is None:
        nrows = 2 if histogram_bins else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=1)
    axes = np.atleast_1d(axes)
    axes[0].eventplot(spiketrains, **kwargs)
    axes[0].set_yticks([0, len(spiketrains) - 1])
    axes[0].set_title(title)

    if histogram_bins:
        axes[1].hist(np.hstack(spiketrains), bins=histogram_bins)
        axes[1].set_ylabel("Spike count")

    axes[-1].set_xlabel(f"Time ({units.dimensionality})")

    if len(axes) == 1:
        axes = axes[0]

    return axes
