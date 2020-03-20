"""
Standard plot function for pairwise unitary event analysis results resembling the original publication.
"""

import numpy
import quantities as pq
import matplotlib.pyplot as plt
import string
import elephant.unitary_event_analysis as ue

plot_params_default = {
    # epochs to be marked on the time axis
    'events': [],
    # save figure
    'save_fig': False,
    # show figure
    'showfig': True,
    # figure size
    'figsize': (10, 12),
    # right margin
    'right': 0.9,
    # top margin
    'top': 0.9,
    # bottom margin
    'bottom': 0.1,
    # left margin
    'left': 0.1,
    # horizontal white space between subplots
    'hspace': 0.5,
    # width white space between subplots
    'wspace': 0.5,
    # font size
    'fsize': 12,
    # the actual unit ids from the experimental recording
    'unit_real_ids': ['not specified', 'not specified'],
    # line width
    'lw': 2,
    # y limit for the surprise
    'S_ylim': (-3, 3),
    # marker size for the UEs and coincidences
    'ms': 5,
    # path and file name for saving the figure
    'path_filename_format': 'figure.pdf'
}


def plot_ue(data, Js_dict, sig_level, binsize, winsize, winstep,
            pattern_hash, plot_params_user):
    """
    Plots the results of pairwise unitary event analysis as a column of six subplots,
    comprised of raster plot, peri-stimulus time histogram, coincident event plot,
    coincidence rate plot, significance plot and unitary event plot, respectively.

    Parameters
    ----------
    data : list of list of neo.SpikeTrain
        A nested list of trials, neurons and their neo.SpikeTrain objects, respectively.
        This should be identical to the one used to generate Js_dict.
    Js_dict : dict
        The output of elephant.unitary_event_analysis.jointJ_window_analysis function.
        The values of each key has the shape of
             different pattern hash --> 0-axis
             different window --> 1-axis
        Js: list of float
          JointSurprise of different given patterns within each window.
        indices: list of list of int
          A list of indices of pattern within each window.
        n_emp: list of int
          The empirical number of each observed pattern.
        n_exp: list of float
          The expected number of each pattern.
        rate_avg: list of float
          The average firing rate of each neuron.
    sig_level : float
        The significance threshold used to determine which coincident events are
        classified as unitary events within a window.
    binsize : quantities.Quantity
        The size of bins for discretizing spike trains. This value should be identical
        to the one used to generate Js_dict.
    winsize : quantities.Quantity
        The size of the window of analysis. This value should be identical to the one
        used to generate Js_dict.
    winstep : quantities.Quantity
        The size of the window step. This value should be identical to the one used to
        generate Js_dict.
    pattern_hash : list of int
        List of interested patterns in hash values. This value should be identical to
        the one used to generate Js_dict
    plot_params_user : dict
        A dictionary of plotting parameters used to update the default plotting
        parameter values.
        events : list
            Epochs to be marked on the time axis.
        savefig : bool
            Saves the figure to disk if True.
        showfig : bool
            Displays the figure on screen if True.
        figsize : tuple of int
            The dimensions for the figure size.
        right : float
            The size of the right margin.
        top : float
            The size of the top margin.
        bottom : float
            The size of the bottom margin.
        left : float
            The size of the left margin.
        hspace : flaot
            The size of the horizontal white space between subplots.
        wspace : float
            The width of the white space between subplots.
        fsize : int
            The size of the font.
        unit_real_ids : list of int
            The unit ids from the experimental recording.
        lw : int
            The default line width.
        S_ylim : tuple of ints or floats
            The y-axis limits for the joint surprise plot.
        ms : int,
            The marker size for the unitary events and coincidences.
        path_filename_format: str
            The path and the filename to save the figure. The format is inferred
            from the filename extension.
    """

    t_start = data[0][0].t_start
    t_stop = data[0][0].t_stop

    N = len(data[0])

    t_winpos = ue._winpos(t_start, t_stop, winsize, winstep)
    Js_sig = ue.jointJ(sig_level)
    num_tr = len(data)

    # figure format
    plot_params = plot_params_default
    plot_params.update(plot_params_user)
    globals().update(plot_params)
    if len(unit_real_ids) != N:
        raise ValueError('length of unit_ids should be' +
                         'equal to number of neurons!')
    plt.rcParams.update({'font.size': fsize})
    plt.rc('legend', fontsize=fsize)

    num_row, num_col = 6, 1
    ls = '-'
    alpha = 0.5
    plt.figure(1, figsize=figsize)
    if 'suptitle' in plot_params.keys():
        plt.suptitle("Trial aligned on " +
                     plot_params['suptitle'], fontsize=20)
    plt.subplots_adjust(top=top, right=right, left=left,
                        bottom=bottom, hspace=hspace, wspace=wspace)

    print('plotting raster plot ...')
    ax0 = plt.subplot(num_row, 1, 1)
    ax0.set_title('Spike Events')
    for n in range(N):
        for tr, data_tr in enumerate(data):
            ax0.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     '.', markersize=0.5, color='k')
        if n < N - 1:
            ax0.axhline((tr + 2) * (n + 1), lw=lw, color='k')
    ax0.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax0.set_yticks([num_tr + 1, 2*num_tr + 1])
    ax0.set_yticklabels([1, num_tr+1], fontsize=fsize)
    ax0.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax0.set_ylabel('Trial', fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            ax0.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)
    Xlim = ax0.get_xlim()
    ax0.text(Xlim[1] - 200, -24, f'Unit {unit_real_ids[0]}')
    ax0.text(Xlim[1] - 200, num_tr * 2 + 7, f'Unit {unit_real_ids[1]}')

    print('plotting Spike Rates ...')
    ax1 = plt.subplot(num_row, 1, 2, sharex=ax0)
    ax1.set_title('Spike Rates')
    for n in range(N):
        ax1.plot(t_winpos + winsize / 2.,
                 Js_dict['rate_avg'][:, n].rescale('Hz'),
                 label='Unit ' + str(unit_real_ids[n]), lw=lw)
    ax1.set_ylabel('(1/s)', fontsize=fsize)
    ax1.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    max_val_psth = 40
    ax1.set_ylim(0, max_val_psth)
    ax1.set_yticks([0, int(max_val_psth / 2), int(max_val_psth)])
    ax1.legend(fontsize=fsize//2)
    for key in events.keys():
        for e_val in events[key]:
            ax1.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)

    print('plotting Raw Coincidences ...')
    ax2 = plt.subplot(num_row, 1, 3, sharex=ax0)
    ax2.set_title('Coincident Events')
    for n in range(N):
        for tr, data_tr in enumerate(data):
            ax2.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1,
                     '.', markersize=0.5, color='k')
            ax2.plot(
                numpy.unique(Js_dict['indices']['trial' + str(tr)]) *
                binsize,
                numpy.ones_like(numpy.unique(Js_dict['indices'][
                    'trial' + str(tr)])) * tr + n * (num_tr + 1) + 1,
                ls='', ms=ms, marker='s', markerfacecolor='none',
                markeredgecolor='c')
        if n < N - 1:
            ax2.axhline((tr + 2) * (n + 1), lw=lw, color='k')
    ax2.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax2.set_yticks([num_tr + 1, 2*num_tr + 1])
    ax2.set_yticklabels([1, num_tr+1], fontsize=fsize)
    ax2.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax2.set_ylabel('Trial', fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            ax2.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)

    print('plotting emp. and exp. coincidences rate ...')
    ax3 = plt.subplot(num_row, 1, 4, sharex=ax0)
    ax3.set_title('Coincidence Rates')
    ax3.plot(t_winpos + winsize / 2.,
             Js_dict['n_emp'] / (winsize.rescale('s').magnitude * num_tr),
             label='Empirical', lw=lw, color='c')
    ax3.plot(t_winpos + winsize / 2.,
             Js_dict['n_exp'] / (winsize.rescale('s').magnitude * num_tr),
             label='Expected', lw=lw, color='m')
    ax3.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax3.set_ylabel('(1/s)', fontsize=fsize)
    ax3.legend(fontsize=fsize//2)
    YTicks = ax3.get_ylim()
    ax3.set_yticks([0, YTicks[1] / 2, YTicks[1]])
    for key in events.keys():
        for e_val in events[key]:
            ax3.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)

    print('plotting Surprise ...')
    ax4 = plt.subplot(num_row, 1, 5, sharex=ax0)
    ax4.set_title('Statistical Significance')
    ax4.plot(t_winpos + winsize / 2., Js_dict['Js'], lw=lw, color='k')
    ax4.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax4.axhline(Js_sig, ls='-', color='r')
    ax4.axhline(-Js_sig, ls='-', color='g')
    ax4.text(t_winpos[30], Js_sig + 0.3, '$\\alpha +$', color='r')
    ax4.text(t_winpos[30], -Js_sig - 0.5, '$\\alpha -$', color='g')
    ax4.set_yticks([ue.jointJ(0.99), ue.jointJ(0.5), ue.jointJ(0.01)])
    ax4.set_yticklabels([0.99, 0.5, 0.01])

    ax4.set_ylim(S_ylim)
    for key in events.keys():
        for e_val in events[key]:
            ax4.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)

    print('plotting UEs ...')
    ax5 = plt.subplot(num_row, 1, 6, sharex=ax0)
    ax5.set_title('Unitary Events')
    for n in range(N):
        for tr, data_tr in enumerate(data):
            ax5.plot(data_tr[n].rescale('ms').magnitude,
                     numpy.ones_like(data_tr[n].magnitude) *
                     tr + n * (num_tr + 1) + 1, '.',
                     markersize=0.5, color='k')
            sig_idx_win = numpy.where(Js_dict['Js'] >= Js_sig)[0]
            if len(sig_idx_win) > 0:
                x = numpy.unique(Js_dict['indices']['trial' + str(tr)])
                if len(x) > 0:
                    xx = []
                    for j in sig_idx_win:
                        xx = numpy.append(xx, x[numpy.where(
                            (x * binsize >= t_winpos[j]) &
                            (x * binsize < t_winpos[j] + winsize))])
                    ax5.plot(
                        numpy.unique(
                            xx) * binsize,
                        numpy.ones_like(numpy.unique(xx)) *
                        tr + n * (num_tr + 1) + 1,
                        ms=ms, marker='s', ls='', mfc='none', mec='r')
        if n < N - 1:
            ax5.axhline((tr + 2) * (n + 1), lw=lw, color='k')
    ax5.set_yticks([num_tr + 1, 2*num_tr + 1])
    ax5.set_yticklabels([1, num_tr+1], fontsize=fsize)
    ax5.set_ylim(0, (tr + 2) * (n + 1) + 1)
    ax5.set_xlim(0, (max(t_winpos) + winsize).rescale('ms').magnitude)
    ax5.set_ylabel('Trial', fontsize=fsize)
    ax5.set_xlabel(f'Time ({t_start.dimensionality.string})', fontsize=fsize)
    for key in events.keys():
        for e_val in events[key]:
            ax5.axvline(e_val, ls=ls, color='r', lw=lw, alpha=alpha)
            ax5.text(e_val - 10 * pq.ms,
                     S_ylim[0] - 35, key, fontsize=fsize, color='r')

    for i in range(num_row):
        ax = locals()['ax' + str(i)]
        ax.text(-0.05, 1.1, string.ascii_uppercase[i],
                transform=ax.transAxes, size=fsize + 5,
                weight='bold')
    if save_fig:
        plt.savefig(path_filename_format)
        if not showfig:
            plt.cla()
            plt.close()

    if showfig:
        plt.show()
