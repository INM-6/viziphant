"""
Simple plotting functions visualizing the output of the Gaussian process
factor analysis.
"""

import neo
import itertools
import warnings
from collections import defaultdict
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from elephant.conversion import BinnedSpikeTrain


def plot_cumulative_explained_variance(loading_matrix):
    """
    This function plots the cumulative explained variance. It allows
    to visually identify an appropriate number of dimensions which is
    small on the one hand, but explains a substantial part of the variance
    in the data on the other hand.

    Parameters
    ----------
    loading_matrix : np.ndarray
        The loading matrix defines the mapping between neural space and
        latent state space. It is obtained by fitting a GPFA model and
        stored in the class GPFA.params_estimated['C'] or if orthonomalized
        GPFA.params_estimated['Corth']

    Returns
    -------
    ax : matplotlib.axes.Axes

    """
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
    """
    This function visualizes the loading matrix as a heatmap.

    Parameters
    ----------
    loading_matrix : np.ndarray
        The loading matrix defines the mapping between neural space and
        latent state space. It is obtained by fitting a GPFA model and
        stored in the class GPFA.params_estimated['C'] or if orthonomalized
        GPFA.params_estimated['Corth']

    Returns
    -------
    ax : matplotlib.axes.Axes

    """

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


def plot_single_dimension_vs_time(returned_data,
                                  gpfa_instance,
                                  dimension_index=0,
                                  orthonomalized_dimensions=True,
                                  n_trials_to_plot=20,
                                  trial_grouping_dict=None,
                                  colors=['grey'],
                                  plot_single_trajectories=True,
                                  plot_group_averages=False,
                                  ax=None,
                                  plot_args_single={'linewidth': 0.3,
                                                    'alpha': 0.4,
                                                    'linestyle': '-'},
                                  plot_args_average={'linewidth': 2,
                                                     'alpha': 1,
                                                     'linestyle': 'dashdot'}):

    """
    This function plots one latent space state dimension versus time.

    Optional visual aids are offered such as grouping the trials and color
    coding their traces.
    Changes to optics of the plot can be applied by providing respective
    dictionaries.

    This function is an adaption of the MATLAB implementation
    by Byron Yu which was published with his paper:
    Yu et al., J Neurophysiol, 2009.

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
        `returned_data`.
    dimension_index : int
        Index of the dimension to plot (0 < dimension_index < dimensionality).
    orthonomalized_dimensions : bool
        Boolean which specifies whether to plot the orthonomalized latent
        state space dimension corresponding to the entry 'xorth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'xsm' (False).
        Beware that the unconstrained state space dimensions 'xsm' are not
        ordered by their explained variance. These dimensions each represent
        one Gaussian process timescale $\tau$.
        On the contrary, the orthonomalized dimensions 'xorth' are ordered by
        decreasing explained variance, allowing a similar intuitive
        interpretation to the dimensions obtained in a PCA. Due to the
        orthonmalization, these dimensions reflect mixtures of timescales.
    n_trials_to_plot : int
        Number of single trial trajectories to plot.
        Default: 20
    trial_grouping_dict : dict
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the group name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
    colors : list
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict.
        Default: ['grey']
    plot_single_trajectories : bool
        If True, single trial trajectories are plotted.
        Default: True
    plot_group_averages : bool
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
        Default: False
    ax : matplotlib axis or None (default)
        The axis onto which to plot. If None a new figure is created.
        When an axis is given, the function can't handle the figure settings.
        Therefore it is recommended to call seaborn.set() with your preferred
        settings before creating your matplotlib figure in order to control
        your plotting layout.
    plot_args_single : dict
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_average : dict
        Arguments dictionary passed to ax.plot() of the average trajectories.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Example
    -------
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
    >>> gpfa_plots.plot_single_dimension_vs_time(
    ...     returned_data=results,
    ...     gpfa_instance=gpfa,
    ...     dimension_index=0,
    ...     orthonomalized_dimensions=False,
    ...     trial_grouping_dict=trial_grouping_dict,
    ...     colors=[f'C{i}' for i in range(len(trial_grouping_dict))],
    ...     n_trials_to_plot=50)

    """

    single_plot = False
    if ax is None:
        single_plot = True
        fig, ax = plt.subplots()

    data = _check_input_data(returned_data, orthonomalized_dimensions)
    colors = _check_colors(colors, trial_grouping_dict)
    # infer n_trial from shape of the data
    n_trials = data.shape[0]
    # infer n_time_bins from maximal number of bins
    n_time_bins = gpfa_instance.transform_info['num_bins'].max()

    # initialize buffer dictionary to handle averages of grouped trials
    if trial_grouping_dict:
        data_buffer = defaultdict(list)

    # loop over trials
    for trial_idx in range(min(n_trials, n_trials_to_plot)):
        dat = data[trial_idx]

        trial_type = _get_trial_type(trial_grouping_dict, trial_idx)
        color = colors[list(trial_grouping_dict.keys()).index(trial_type)]

        # plot single trial trajectories
        if plot_single_trajectories:
            ax.plot(np.arange(1, n_time_bins + 1),
                    dat[dimension_index, :],
                    color=color,
                    label=trial_type,
                    **plot_args_single)

    if plot_group_averages and trial_grouping_dict:
        for trial_idx in range(n_trials):
            dat = data[trial_idx]
            trial_type = _get_trial_type(trial_grouping_dict, trial_idx)

            # fill buffer dictionary to handle averages of grouped trials
            if trial_type is not None:
                data_buffer[trial_type].append(dat)

        for i_group, (trial_type,
                      group_data_buffer) in enumerate(data_buffer.items()):

            group_average = np.mean(group_data_buffer, axis=0)

            ax.plot(np.arange(1, n_time_bins + 1),
                    group_average[dimension_index],
                    color=colors[i_group],
                    label=trial_type,
                    **plot_args_average)

    _set_title_dimensions_vs_time(ax,
                                  dimension_index,
                                  orthonomalized_dimensions,
                                  data,
                                  gpfa_instance)

    x_axis_ticks, x_axis_ticks_lengths, y_axis_ticks = \
        _get_axis_limits_and_ticks(data, gpfa_instance, n_time_bins)

    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels(x_axis_ticks_lengths)
    ax.set_yticks(y_axis_ticks)
    ax.set_yticklabels(y_axis_ticks)
    ax.set_xlabel('Time (ms)')

    if single_plot:
        # only plot unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

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
                           plot_args_single={'linewidth': 0.3,
                                             'alpha': 0.4,
                                             'linestyle': '-'},
                           plot_args_average={'linewidth': 2,
                                              'alpha': 1,
                                              'linestyle': 'dashdot'},
                           figure_args={'figsize': (10, 10)},
                           gridspec_args={}):

    """
    This function plots all latent space state dimensions versus time.
    It is a wrapper for the function plot_single_dimension_vs_time and
    places the single plot onto a grid.

    Optional visual aids are offered such as grouping the trials and color
    coding their traces.
    Changes to optics of the plot can be applied by providing respective
    dictionaries.

    This function is an adaption of the MATLAB implementation
    by Byron Yu which was published with his paper:
    Yu et al., J Neurophysiol, 2009.

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
        `returned_data`.
    dimension_index : int
        Index of the dimension to plot (0 < dimension_index < dimensionality).
    orthonomalized_dimensions : bool
        Boolean which specifies whether to plot the orthonomalized latent
        state space dimension corresponding to the entry 'xorth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'xsm' (False).
        Beware that the unconstrained state space dimensions 'xsm' are not
        ordered by their explained variance. These dimensions each represent
        one Gaussian process timescale $\tau$.
        On the contrary, the orthonomalized dimensions 'xorth' are ordered by
        decreasing explained variance, allowing a similar intuitive
        interpretation to the dimensions obtained in a PCA. Due to the
        orthonmalization, these dimensions reflect mixtures of timescales.
    n_trials_to_plot : int
        Number of single trial trajectories to plot.
        Default: 20
    trial_grouping_dict : dict
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the group name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
    colors : list
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict.
        Default: ['grey']
    plot_single_trajectories : bool
        If True, single trial trajectories are plotted.
        Default: True
    plot_group_averages : bool
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
        Default: False
    n_columns : int
        Number of columns of the grid onto which the single plots are placed.
        The number of rows are deduced from the number of dimensions
        to be plotted.
    ax : matplotlib axis or None (default)
        The axis onto which to plot. If None a new figure is created.
        When an axis is given, the function can't handle the figure settings.
        Therefore it is recommended to call seaborn.set() with your preferred
        settings before creating your matplotlib figure in order to control
        your plotting layout.
    plot_args_single : dict
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_average : dict
        Arguments dictionary passed to ax.plot() of the average trajectories.
    figure_args : dict
        Arguments dictionary passed to matplotlib.pyplot.figure(),
        if ax is None.
    gridspec_args : dict
        Arguments dictionary passed to matplotlib.gridspec.GridSpec().

    Returns
    -------
    ax : matplotlib.axes.Axes

    Example
    -------
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
    >>> gpfa_plots.plot_dimension_vs_time(
    ...     returned_data=results,
    ...     gpfa_instance=gpfa,
    ...     orthonomalized_dimensions=True,
    ...     trial_grouping_dict=trial_grouping_dict,
    ...     colors=[f'C{i}' for i in range(len(trial_grouping_dict))],
    ...     n_columns=3,
    ...     n_trials_to_plot=50)

    """

    f = plt.figure(**figure_args)

    # deduce n_rows from n_columns
    n_dimensions = gpfa_instance.x_dim
    n_rows = int(np.ceil(n_dimensions / n_columns))

    grid_specification = gridspec.GridSpec(n_rows,
                                           n_columns, **gridspec_args)

    for k, gs in zip(range(n_dimensions), grid_specification):
        ax = plot_single_dimension_vs_time(
            returned_data,
            gpfa_instance,
            dimension_index=k,
            orthonomalized_dimensions=orthonomalized_dimensions,
            n_trials_to_plot=n_trials_to_plot,
            trial_grouping_dict=trial_grouping_dict,
            colors=colors,
            plot_single_trajectories=plot_single_trajectories,
            plot_group_averages=plot_group_averages,
            ax=plt.subplot(gs),
            plot_args_single=plot_args_single,
            plot_args_average=plot_args_average)

        # plot legend only for first subplot
        if k == 0:
            # only plot unique labels
            handles, labels = f.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

    grid_specification.tight_layout(f)

    return f


def plot_trajectories(returned_data,
                      gpfa_instance,
                      block_with_cut_trials=None,
                      neo_event_name=None,
                      relevant_events=None,
                      dimensions_to_plot=[0, 1, 2],
                      orthonomalized_dimensions=True,
                      n_trials_to_plot=20,
                      trial_grouping_dict=None,
                      colors=['grey'],
                      plot_single_trajectories=True,
                      plot_group_averages=False,
                      plot_args_single={'linewidth': 0.3,
                                        'alpha': 0.4,
                                        'linestyle': '-'},
                      plot_args_marker={'alpha': 0.4,
                                        'markersize': 5},
                      plot_args_average={'linewidth': 2,
                                         'alpha': 1,
                                         'linestyle': 'dashdot'},
                      plot_args_marker_start={'marker': 'p',
                                              'markersize': 10,
                                              'label': 'trial_start'}):

    """
    This function allows for 2D and 3D visualization of the latent space
    variables identified by the GPFA.

    Optional visual aids are offered such as grouping the trials and color
    coding their traces.
    Changes to optics of the plot can be applied by providing respective
    dictionaries.

    This function is an adaption of the MATLAB implementation
    by Byron Yu which was published with his paper:
    Yu et al., J Neurophysiol, 2009.

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
    block_with_cut_trials : neo.Block
        The neo.Block should contain each single trial as a separate
        neo.Segment including the neo.Event with a specified
        `neo_event_name`.
    neo_event_name : str
        A string specifying the name of the neo.Event which should be used
        to identify the event times and labels of the `relevant_events`.
    relevant_events : list of str
        List of names of the event labels that should be plotted onto each
        single trial trajectory.
    dimensions_to_plot : list
        List specifying the indices of the dimensions to use for the
        2D or 3D plot.
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
    n_trials_to_plot : int
        Number of single trial trajectories to plot.
        Default: 20
    trial_grouping_dict : dict
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the group name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
    colors : list
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict.
        Default: ['grey']
    plot_single_trajectories : bool
        If True, single trial trajectories are plotted.
        Default: True
    plot_group_averages : bool
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
        Default: False
    plot_args_single : dict
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_marker : dict
        Arguments dictionary passed to ax.plot() for the single trial events.
    plot_args_average : dict
        Arguments dictionary passed to ax.plot() of the average trajectories.
        if ax is None.
    plot_args_marker_start : dict
        Arguments dictionary passed to ax.plot() for the marker of the
        average trajectory start.

    Returns
    -------
    f : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Example
    -------
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
    >>>    gpfa_plots.plot_trajectories(
    ...        results,
    ...        gpfa,
    ...        block_with_cut_trials=None,
    ...        relevant_events=None,
    ...        dimensions_to_plot=[0,1,2],
    ...        trial_grouping_dict=trial_grouping_dict,
    ...        plot_group_averages=False,
    ...        plot_single_trajectories=True,
    ...        n_trials_to_plot=200,
    ...        plot_args_single={'linewidth': 0.8,
    ...                          'alpha': 0.4, 'linestyle': '-'})

    """
    # prepare the input
    projection, dimensions = \
        _check_dimensions(gpfa_instance, dimensions_to_plot)
    X = _check_input_data(returned_data, orthonomalized_dimensions)
    colors = _check_colors(colors, trial_grouping_dict)

    # infer n_trial from shape of the data
    n_trials = X.shape[0]

    # infer n_time_bins from maximal number of bins
    n_time_bins = gpfa_instance.transform_info['num_bins'].max()

    # initialize figure and axis
    f = plt.figure()
    ax = f.gca(projection=projection, aspect='auto')

    # initialize buffer dictionary to handle averages of grouped trials
    if trial_grouping_dict:
        data_buffer = {}
        for i_group, (trial_type,
                      trial_ids) in enumerate(trial_grouping_dict.items()):
            data_buffer[trial_type] = np.zeros((dimensions,
                                                n_time_bins))

    # loop over trials
    for trial_idx in range(min(n_trials, n_trials_to_plot)):
        dat = X[trial_idx][dimensions_to_plot, :]
        trial_type = _get_trial_type(trial_grouping_dict, trial_idx)
        color = colors[list(trial_grouping_dict.keys()).index(trial_type)]

        if plot_single_trajectories:
            if dimensions == 2:
                ax.plot(dat[0], dat[1],
                        color=color,
                        label=trial_type,
                        **plot_args_single)
            elif dimensions == 3:
                ax.plot(dat[0], dat[1], dat[2],
                        color=color,
                        label=trial_type,
                        **plot_args_single)

            # plot single trial events
            if block_with_cut_trials and neo_event_name and relevant_events:
                time_bins_with_relevant_event, relevant_event_labels = \
                    _get_event_times_and_labels(block_with_cut_trials,
                                                trial_idx,
                                                neo_event_name,
                                                relevant_events,
                                                gpfa_instance)

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
                                color=color,
                                **plot_args_marker)
                    elif dimensions == 3:
                        ax.plot([dat[0][event_time]],
                                [dat[1][event_time]],
                                [dat[2][event_time]],
                                marker=next(marker),
                                label=event_label,
                                color=color,
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
                        color=colors[i_group],
                        **plot_args_marker_start)

    _set_axis_labels_trajectories(ax,
                                  orthonomalized_dimensions,
                                  dimensions_to_plot)

    # only plot unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

    return f, ax


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


def _get_axis_limits_and_ticks(data, gpfa_instance, n_time_bins):

    # stack list of arrays
    data_stacked = np.stack(data)

    # prepare ticks
    x_axis_ticks_step = np.ceil(n_time_bins / 25.) * 5

    x_axis_ticks = np.arange(1, n_time_bins + 1, x_axis_ticks_step)

    x_axis_ticks_lengths = np.arange(
        0, (n_time_bins * gpfa_instance.bin_size).rescale(pq.ms).magnitude,
        (x_axis_ticks_step * gpfa_instance.bin_size).rescale(pq.ms).magnitude,
        dtype=np.int)

    # round max value to next highest 1e-1
    y_max = np.ceil(10 * np.abs(data_stacked).max()) / 10

    y_axis_ticks = [-y_max, 0, y_max]

    return x_axis_ticks, x_axis_ticks_lengths, y_axis_ticks


def _set_title_dimensions_vs_time(ax,
                                  latent_variable_idx,
                                  orthonomalized_dimensions,
                                  data,
                                  gpfa_instance):
    if orthonomalized_dimensions:
        str = r'$\tilde{{\mathbf{{x}}}}_{{{},:}}$'.format(latent_variable_idx)

        # percentage of variance of the dimensionality reduced data
        # that is explained by this latent variable
        variances = [np.var(np.hstack(data)[i, :]) for
                     i in range(gpfa_instance.x_dim)]
        total_variance = np.sum(variances)
        explained_variance = np.round(
            variances[latent_variable_idx]/total_variance*100, 2)

        title = str + f'% exp. var.: {explained_variance} %'
    else:
        str = r"${{\mathbf{{x}}}}_{{{},:}}$".format(latent_variable_idx)

        # time scale of the gaussian process associated to this latent variable
        GP_time_scale = np.round(gpfa_instance.bin_size / \
                                 np.sqrt(gpfa_instance.params_estimated['gamma'][latent_variable_idx]), 2)

        title = str + rf'$\tau$: {GP_time_scale}'

    ax.set_title(title, fontsize=16)


def _set_axis_labels_trajectories(ax,
                                  orthonomalized_dimensions,
                                  dimensions_to_plot):
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
                                neo_event_name,
                                relevant_events,
                                gpfa_instance):

    trial_events = block_with_cut_trials.segments[trial_idx].filter(
        objects='Event',
        name=neo_event_name)[0]

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
