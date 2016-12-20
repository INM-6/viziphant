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

def rasterplot(ax, neo_obj,
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

    segments = neo_obj.read_block().segments
    for seg_count, seg in enumerate(segments):
        tmin = min([st.t_start for st in seg.spiketrains])
        tmax = min([st.t_stop for st in seg.spiketrains])
        ids = [st.annotations['id'] for st in seg.spiketrains]
        minid = min(ids)
        maxid = max(ids)
        for st_count, st in enumerate(seg.spiketrains):
            # sns.stripplot(st.times.magnitude,
            #               [st.annotations['id']] * st.__len__(),
            #               orient='h')

            ax.plot(st.times.magnitude, [st.annotations['id']] * st.__len__(),
                    markertype, ms=markersize)

        # sns.plt.show()
        # y = np.concatenate(np.array(
        #     [[st.annotations['id']] * st.__len__() for st in seg.spiketrains]))
        # x = np.concatenate(
        #     np.array([st.tmes.magnitude for st in seg.spiketrains]))
        # g = sns.JointGrid(x,y)
        # g.plot(sns.stripplot,sns.distplot)

        axhisty.barh(np.array([st.annotations['id'] for st in seg.spiketrains]),
                     np.array([np.sum(st.times) for st in seg.spiketrains]))
        axhistx.hist(np.concatenate(seg.spiketrains), bins)
        period = tmax - tmin
        ax.set_xlim(tmin - .01 * period, tmax + .01 * period)
        id_span = maxid - minid
        ax.set_ylim(minid - .01 * id_span, maxid + .01 * id_span)
        axhistx.set_xlim(ax.get_xlim())
        axhisty.set_ylim(ax.get_ylim())
        ax.set_xlabel('t [{0}]'.format(seg.spiketrains[0].units.dimensionality))
        ax.set_ylabel('Unit ID')
        axhistx.get_xaxis().set_visible(False)
        axhistx.get_yaxis().set_visible(False)
        axhisty.get_xaxis().set_visible(False)
        axhisty.get_yaxis().set_visible(False)

        plt.draw()

    return ax, axhistx, axhisty
