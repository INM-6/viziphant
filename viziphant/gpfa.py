"""
Simple plotting functions visualizing the output of the Gaussian process
factor analysis.
"""

import numpy as np
import itertools
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from elephant.conversion import BinnedSpikeTrain
import warnings
import neo
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_cumulative_explained_variance(loading_matrix):
    eigenvalues = np.linalg.eigvals(np.dot(loading_matrix.transpose(),
                                           loading_matrix))
    total_variance = np.sum(eigenvalues)

    # sort by decreasing variance explained
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[-1::-1]
    cumulative_variance = np.cumsum(sorted_eigenvalues / total_variance)

    fig, ax = plt.subplots()
    ax.plot(cumulative_variance, 'o-')

    ax.grid()
    ax.set_title('Eigenspectrum of estimated shared covariance matrix')
    ax.set_xlabel('Latent Dimensionality')
    ax.set_ylabel('Cumulative % of shared variance explained')

    return ax


def plot_loading_matrix(loading_matrix):

    fig, ax = plt.subplots()

    heatmap = ax.imshow(loading_matrix,
                        aspect='auto',
                        interpolation='none')

    ax.set_title('Loading Matrix')
    ax.set_ylabel('Neuron ID')
    ax.set_xlabel('Latent Variable')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = plt.colorbar(heatmap, cax=cax)
    colorbar.set_label('Latent Variable Weight')

    return ax


def plot_dimension_vs_time(returned_data,
                           gpfa_instance,
                           orthonomalized_dimensions=True,
                           n_trials_to_plot=20,
                           trial_grouping_dict=None,
                           colors=['grey'],
                           plot_single_trajectories=True,
                           plot_group_averages=False,
                           n_columns=4,
                           **extraOpts):

    """
    This function plots each latent space state dimension versus time
    in a seperate panel each.

    Optional visual aids are offered such as sorting, grouping and color coding
    on the basis of the arrangement in list of spike trains and spike train
    annotations.
    Changes to optics of the dot marker, the separators and the legend can be
    applied by providing a dict with the respective parameters. Changes and
    additions to the dot display itself or the two histograms are best realized
    by using the returned axis handles.

    This function was adapted from the original MATLAB implementation of the
    plotEachDimVsTime by Byron Yu which was published with his
    paper Yu et al., J Neurophysiol, 2009.


    Parameters
    ----------
    returned_data : np.ndarray or dict
        When the length of `returned_data` is one, a single np.ndarray,
        containing the requested data (the first entry in `returned_data`
        keys list), is returned. Otherwise, a dict of multiple np.ndarrays
        with the keys identical to the data names in `returned_data` is
        returned.

        N-th entry of each np.ndarray is a np.ndarray of the following
        shape, specific to each data type, containing the corresponding
        data for the n-th trial:

            `xorth`: (#latent_vars, #bins) np.ndarray

            `xsm`:  (#latent_vars, #bins) np.ndarray

            `y`:  (#units, #bins) np.ndarray

            `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

            `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

        Note that the num. of bins (#bins) can vary across trials,
        reflecting the trial durations in the given `spiketrains` data.
    gpfa_instance : class
        Instance of the GPFA() class in elephant, which was used to obtain
        returned_data.
    orthonomalized_dimensions : bool
        Boolean which specifies whether to plot the orthonomalized latent
        state space dimensions corresponding to the entry 'xorth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'xsm' (False).
        Beware that the unconstrained state space dimensions 'xsm' are not
        ordered by their explained variance. These dimensions each represent
        one Gaussian process timescale.
        On the contrary, the orthonomalized dimensions 'xorth' are ordered by
        decreasing explained variance, allowing a similar intuitive
        interpretation to the dimensions obtained in a PCA. Due to the
        orthonmalization, these dimensions reflect mixtures of timescales.
    plot_single_trajectories : bool
        If True, single trial trajectories are plotted.
    n_trials_to_plot : int
        Number of single trial trajectories to plot.
    plot_group_averages : bool
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
    trial_grouping_dict : dict
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the groups name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
    colors : list
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict.
        Default: ['grey']
    n_columns : int
        Specifies the number of columns to plot the dimensions in.


    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    independent Poisson spike trains recorded in 50 trials with randomized
    rates up to 100 Hz and plot the resulting orthonomalized latent state
    space dimensions.

    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa import GPFA
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> data = []
    >>> for trial in range(50):
    >>>     n_channels = 20
    >>>     firing_rates = np.random.randint(low=1, high=100,
    ...                                      size=n_channels) * pq.Hz
    >>>     spike_times = [homogeneous_poisson_process(rate=rate)
    ...                    for rate in firing_rates]
    >>>     data.append((trial, spike_times))
    ...
    >>> gpfa = GPFA(bin_size=20*pq.ms, x_dim=8)
    >>> gpfa.fit(data)
    >>> results = gpfa.transform(data, returned_data=['xorth', 'xsm'])

    >>> trial_id_lists = np.arange(50).reshape(5,10)
    >>> trial_group_names = ['A', 'B', 'C', 'D', 'E']
    >>> trial_grouping_dict = {}
    >>> for trial_group_name, trial_id_list in zip(trial_group_names,
    ...                                            trial_id_lists):
    >>>     trial_grouping_dict[trial_group_name] = trial_id_list
    ...
    >>>

    """

    # TODO: remove hardcoded parameters
    # TODO: modularize -> function for single dimension -> compose within gridspec
    # TODO: pep8
    # TODO: use unique axes objects to remove the following warning
    """
    MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance
    """

    f = plt.figure(figsize=(10, 10))

    X = _check_input_data(returned_data, orthonomalized_dimensions)
    colors = _check_colors(colors, trial_grouping_dict)

    # convert array of 2d arrays into 3d array
    X_3D = np.stack(X)

    # round max value to next highest 1e-1
    X_max = np.ceil(10 * np.abs(X_3D).max()) / 10

    T_max = gpfa_instance.transform_info['num_bins'].max()

    # prepare ticks
    x_axis_ticks_step = np.ceil(T_max / 25.) * 5

    print(T_max, x_axis_ticks_step)

    x_axis_ticks = np.arange(1, T_max + 1, x_axis_ticks_step)

    x_axis_ticks_lengths = np.arange(0, (T_max * gpfa_instance.bin_size).rescale(pq.ms).magnitude,
                                     (x_axis_ticks_step * gpfa_instance.bin_size).rescale(pq.ms).magnitude,
                                     dtype=np.int)
    y_axis_ticks = [-X_max, 0, X_max]

    # deduce n_rows from n_columns
    n_rows = int(np.ceil(X_3D.shape[1] / n_columns))

    # loop over trials
    for n in range(min(X.shape[0], n_trials_to_plot)):
        dat = X[n]
        T = gpfa_instance.transform_info['num_bins'][n]

        # loop over latent variables
        for k in range(dat.shape[0]):

            # initalize subplot
            ax = plt.subplot(n_rows, n_columns, k + 1)

            if trial_grouping_dict:
                for i_group, (trial_type, trial_ids) in enumerate(trial_grouping_dict.items()):
                    if n in trial_ids:
                        col = colors[i_group]
                        lw = 0.5
                        if plot_group_averages:
                            alpha = 0.1
                        else:
                            alpha = 0.8
                        break
            else:
                col = 'gray'
                lw = 0.5
                alpha = 0.8
                trial_type = None

            if plot_single_trajectories:
                plt.plot(np.arange(1, T + 1),
                         dat[k, :],
                         linewidth=lw,
                         color=col,
                         alpha=alpha, label=trial_type)

    if plot_group_averages and trial_grouping_dict:
        for i_group, (trial_type, trial_ids) in enumerate(trial_grouping_dict.items()):
            data_buffer = []
            for trial_id in trial_ids:
                data_buffer.append(X[trial_id])

            group_average = np.mean(data_buffer, axis=0)
            # group_std = np.std(data_buffer, axis=0)

            # loop over latent variables
            for k in range(dat.shape[0]):
                ax = plt.subplot(n_rows, n_columns, k + 1)
                plt.plot(np.arange(1, T + 1), group_average[k],
                         linewidth=1, color=colors[i_group], label=trial_type)

    # loop over latent variables
    for k in range(dat.shape[0]):
        ax = plt.subplot(n_rows, n_columns, k + 1)
        plt.axis([1, T_max, 1.1 * min(y_axis_ticks), 1.1 * max(y_axis_ticks)])

        # by default returned_data is an array containing the orthonormalized posterior mean of latent variable
        if isinstance(returned_data, np.ndarray):
            legend_string = r'$\tilde{{\mathbf{{x}}}}_{{{},:}}$'.format(k)
        elif isinstance(returned_data, dict):
            if orthonomalized_dimensions:
                legend_string = r'$\tilde{{\mathbf{{x}}}}_{{{},:}}$'.format(k)
            else:
                if 'xsm' in returned_data.keys():
                    legend_string = r"${{\mathbf{{x}}}}_{{{},:}}$".format(k)

        plt.title(legend_string, fontsize=16)

        plt.xticks(x_axis_ticks, x_axis_ticks_lengths)
        plt.yticks(y_axis_ticks, y_axis_ticks)
        plt.xlabel('Time (ms)')

        # plot legend only for first subplot
        if k == 0:
            # only plot unique labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()


# TODO: remove hardcoded linewidths etc...
# TODO: consistent namings with plot_dimension_vs_time
# TODO: document
def plot_trajectories(returned_data,
                      gpfa_instance,
                      block_with_cut_trials=None,
                      relevant_events=None,
                      dimensions_to_plot=[0, 1, 2],
                      orthonomalized_dimensions=True,
                      n_trials_to_plot=20,
                      trial_grouping_dict=None,
                      colors=['grey'],
                      plot_single_trajectories=True,
                      plot_group_averages=False,
                      **extraOpts):

    """
    adapted from plot3D @ 2009 Byron Yu -- byronyu@stanford.edu
    # TODO: remove hardcoded linewidths etc...
    """

    # prepare the input
    projection, dimensions = \
        _check_dimensions(gpfa_instance, dimensions_to_plot)
    X = _check_input_data(returned_data, orthonomalized_dimensions)
    colors = _check_colors(colors, trial_grouping_dict)
    n_trials = X.shape[0]

    # Initialize figure and axis
    f = plt.figure()
    ax = f.gca(projection=projection, aspect='auto')

    # initialize buffer dictionary to handle averages of grouped trials
    # infer n_time_bins from first trajectory
    n_time_bins = gpfa_instance.transform_info['num_bins'][0]
    if trial_grouping_dict:
        data_buffer = {}
        for i_group, (trial_type,
                      trial_ids) in enumerate(trial_grouping_dict.items()):
            data_buffer[trial_type] = np.zeros((dimensions,
                                                n_time_bins))

    # loop over trials
    for trial_idx in range(min(n_trials, n_trials_to_plot)):
        dat = X[trial_idx][dimensions_to_plot, :]
        # plot single trial trajectories
        plot_args_single = _set_plot_arguments(trial_grouping_dict,
                                               trial_idx,
                                               colors,
                                               plot_group_averages,
                                               plot='single_trajectories')

        if plot_single_trajectories:
            if dimensions == 2:
                ax.plot(dat[0], dat[1], **plot_args_single)
            elif dimensions == 3:
                ax.plot(dat[0], dat[1], dat[2], **plot_args_single)

            # plot single trial events
            if block_with_cut_trials and relevant_events:
                time_bins_with_relevant_event, relevant_event_labels = \
                    _get_event_times_and_labels(block_with_cut_trials,
                                                trial_idx,
                                                relevant_events,
                                                gpfa_instance)

                plot_args_marker = _set_plot_arguments(trial_grouping_dict,
                                                       trial_idx,
                                                       colors,
                                                       plot_group_averages,
                                                       plot='event_markers')

                marker = itertools.cycle(Line2D.filled_markers)
                for event_id, (event_time,
                               event_label) in enumerate(zip(
                                   time_bins_with_relevant_event,
                                   relevant_event_labels)):
                    if dimensions == 2:
                        ax.plot([dat[0][event_time]],
                                [dat[1][event_time]],
                                marker=next(marker),
                                label=event_label,
                                **plot_args_marker)
                    elif dimensions == 3:
                        ax.plot([dat[0][event_time]],
                                [dat[1][event_time]],
                                [dat[2][event_time]],
                                marker=next(marker),
                                label=event_label,
                                **plot_args_marker)

    if plot_group_averages and trial_grouping_dict:
        for trial_idx in range(n_trials):
            dat = X[trial_idx][dimensions_to_plot, :]
            trial_type = _get_trial_type(trial_grouping_dict, trial_idx)

            # fill buffer dictionary to handle averages of grouped trials
            if trial_type is not None:
                data_buffer[trial_type] += dat

        for i_group, (trial_type,
                      group_data_buffer) in enumerate(data_buffer.items()):

            group_average = group_data_buffer / \
                len(trial_grouping_dict[trial_type])

            plot_args_average = _set_plot_arguments(
                trial_grouping_dict,
                trial_idx,
                colors,
                plot_group_averages,
                plot='average_trajectories')

            if dimensions == 2:
                ax.plot(group_average[0],
                        group_average[1],
                        color=colors[i_group],
                        label=trial_type,
                        **plot_args_average)
                ax.plot([group_average[0][0]],
                        [group_average[1][0]],
                        'p',
                        markersize=10,
                        color=colors[i_group],
                        label='trial_start')
            elif dimensions == 3:
                ax.plot(group_average[0],
                        group_average[1],
                        group_average[2],
                        color=colors[i_group],
                        label=trial_type,
                        **plot_args_average)
                ax.plot([group_average[0][0]],
                        [group_average[1][0]],
                        [group_average[2][0]],
                        'p',
                        markersize=10,
                        color=colors[i_group],
                        label='trial_start')

    _set_axis_labels(ax, orthonomalized_dimensions, dimensions_to_plot)

    # only plot unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()


def _check_input_data(returned_data, orthonomalized_dimensions):

    # by default returned_data is an array containing the
    # orthonormalized posterior mean of latent variable
    if isinstance(returned_data, np.ndarray):
        X = returned_data
    elif isinstance(returned_data, dict):
        if orthonomalized_dimensions:
            X = returned_data['xorth']
        else:
            if 'xsm' in returned_data.keys():
                X = returned_data['xsm']
            else:
                raise ValueError("The latent variables before"
                                 "orthonomalization 'xsm' are not in"
                                 "returned data")

    return X


def _check_colors(colors, trial_grouping_dict):
    if trial_grouping_dict:
        if not len(colors) == len(trial_grouping_dict):
            warnings.warn("Colors per trial group were not specified! "
                          "Employing default matplotlib colors.")
            colors = [f'C{i}' for i in range(len(trial_grouping_dict))]
    return colors


def _check_dimensions(gpfa_instance, dimensions_to_plot):
    # check if enough dimensions are available
    if gpfa_instance.x_dim < len(dimensions_to_plot):
        print("ERROR: Trajectories have less than 3 dimensions.\n")
        return

    # check if enough dimensions are available
    if len(dimensions_to_plot) > 3:
        print("ERROR: Unfortunately it is difficult to visualize "
              "more than 3 dimensions.\n")
        return

    if len(dimensions_to_plot) == 2:
        projection = None
        dimensions = 2
    elif len(dimensions_to_plot) == 3:
        projection = '3d'
        dimensions = 3
    return projection, dimensions


def _get_trial_type(trial_grouping_dict,
                    trial_idx):
    for (trial_type, trial_ids) in trial_grouping_dict.items():
        if trial_idx in trial_ids:
            return trial_type


def _set_plot_arguments(trial_grouping_dict,
                        trial_idx,
                        colors,
                        plot_group_averages,
                        plot):

    if trial_grouping_dict:
        for i_group, (trial_type,
                      trial_ids) in enumerate(trial_grouping_dict.items()):
            if trial_idx in trial_ids:
                col = colors[i_group]
                lw = 0.3
                if plot_group_averages:
                    alpha = 0.4
                else:
                    alpha = 0.8
                break
    else:
        col = 'gray'
        lw = 0.3
        trial_type = None
        alpha = 0.8

    if plot == 'single_trajectories':
        plot_args = {'color': col,
                     'linewidth': lw,
                     'alpha': alpha,
                     'label': trial_type,
                     'linestyle': '-'}

    elif plot == 'event_markers':
        plot_args = {'color': col,
                     'alpha': alpha,
                     'markersize': 5}

    elif plot == 'average_trajectories':
        plot_args = {'linewidth': 1.5,
                     'alpha': 1,
                     'linestyle': 'dashdot'}
    return plot_args


def _set_axis_labels(ax, orthonomalized_dimensions, dimensions_to_plot):
    if orthonomalized_dimensions:
        str1 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(
            dimensions_to_plot[0])
        str2 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(
            dimensions_to_plot[1])
        if len(dimensions_to_plot) == 3:
            str3 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(
                dimensions_to_plot[2])
    else:
        str1 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions_to_plot[0])
        str2 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions_to_plot[1])
        if len(dimensions_to_plot) == 3:
            str3 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions_to_plot[2])
    ax.set_xlabel(str1, fontsize=16)
    ax.set_ylabel(str2, fontsize=16)
    if len(dimensions_to_plot) == 3:
        ax.set_zlabel(str3, fontsize=16)


def _get_event_times_and_labels(block_with_cut_trials,
                                trial_idx,
                                relevant_events,
                                gpfa_instance):

    # TODO: maybe this is too specific to the R2G data
    trial_events = block_with_cut_trials.segments[trial_idx].filter(
        objects='Event',
        name='TrialEvents')[0]

    # get mask for the relevant events
    mask = np.zeros(trial_events.array_annotations['trial_event_labels'].shape,
                    dtype='bool')
    for event in relevant_events:
        mask = np.logical_or(
            mask,
            trial_events.array_annotations['trial_event_labels'] == event)

    # cheating by converting event times to binned spiketrain
    t_start = block_with_cut_trials.segments[trial_idx].t_start
    t_stop = block_with_cut_trials.segments[trial_idx].t_stop

    event_spiketrain = neo.SpikeTrain(trial_events.times[mask],
                                      t_start=t_start,
                                      t_stop=t_stop)
    bin_size = gpfa_instance.bin_size
    binned_event_spiketrain = BinnedSpikeTrain(
        event_spiketrain,
        bin_size=bin_size).to_array().flatten()

    time_bins_with_relevant_event = np.nonzero(binned_event_spiketrain)[0]
    relevant_event_labels = \
        trial_events.array_annotations['trial_event_labels'][[mask]]

    return time_bins_with_relevant_event, relevant_event_labels
