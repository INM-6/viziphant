"""
Gaussian Process Factor Analysis (GPFA) plots
---------------------------------------------

Visualizes transformed trajectories output from
:class:`elephant.gpfa.gpfa.GPFA`

.. autosummary::
    :toctree: toctree/gpfa/

    plot_dimensions_vs_time
    plot_trajectories
    plot_cumulative_explained_variance
    plot_transform_matrix
"""

import itertools
import math
import matplotlib.pyplot as plt
import neo
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from elephant.conversion import BinnedSpikeTrain


def plot_cumulative_shared_covariance(loading_matrix):
    """
    This function plots the cumulative shared covariance. It allows
    to visually identify an appropriate number of dimensions which is
    small on the one hand, but explains a substantial part of the variance
    in the data on the other hand.

    Parameters
    ----------
    loading_matrix : np.ndarray
        The loading matrix defines the mapping between neural space and
        latent state space. It is obtained by fitting a GPFA model and
        stored in ``GPFA.params_estimated['C']`` or if orthonormalized
        ``GPFA.params_estimated['Corth']``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes

    """
    eigenvalues = np.linalg.eigvals(np.dot(loading_matrix.transpose(),
                                           loading_matrix))
    total_variance = np.sum(eigenvalues)

    # sort by decreasing variance explained
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[-1::-1]
    cumulative_variance = np.cumsum(sorted_eigenvalues / total_variance)

    fig, axes = plt.subplots()
    axes.plot(cumulative_variance, 'o-')

    axes.grid()
    axes.set_title('Eigenspectrum of estimated shared covariance matrix')
    axes.set_xlabel('Latent Dimensionality')
    axes.set_ylabel('Cumulative % of shared variance explained')

    return fig, axes


def plot_transform_matrix(loading_matrix, cmap='RdYlGn'):
    """
    This function visualizes the loading matrix as a heatmap.

    Parameters
    ----------
    loading_matrix : np.ndarray
        The loading matrix defines the mapping between neural space and
        latent state space. It is obtained by fitting a GPFA model and
        stored in ``GPFA.params_estimated['C']`` or if orthonormalized
        ``GPFA.params_estimated['Corth']``.
    cmap : str, optional
        Matplotlib imshow colormap.
        Default: 'RdYlGn'

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes

    """

    fig, axes = plt.subplots()

    vmax = np.max(np.abs(loading_matrix))
    vmin = -vmax

    heatmap = axes.imshow(loading_matrix,
                          vmin=vmin, vmax=vmax,
                          aspect='auto',
                          interpolation='none', cmap=cmap)

    axes.set_title('Loading Matrix')
    axes.set_ylabel('Neuron ID')
    axes.set_xlabel('Latent Variable')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = plt.colorbar(heatmap, cax=cax)
    colorbar.set_label('Latent Variable Weight')

    return fig, axes


def plot_dimensions_vs_time(returned_data,
                            gpfa_instance,
                            dimensions='all',
                            orthonormalized_dimensions=True,
                            n_trials_to_plot=20,
                            trial_grouping_dict=None,
                            colors='grey',
                            plot_single_trajectories=True,
                            plot_group_averages=False,
                            n_columns=2,
                            plot_args_single={'linewidth': 0.3,
                                              'alpha': 0.4,
                                              'linestyle': '-'},
                            plot_args_average={'linewidth': 2,
                                               'alpha': 1,
                                               'linestyle': 'dashdot'},
                            figure_args=dict(figsize=(10, 10))):

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

            `latent_variable_orth`: (#latent_vars, #bins) np.ndarray

            `latent_variable`:  (#latent_vars, #bins) np.ndarray

            `y`:  (#units, #bins) np.ndarray

            `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

            `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

        Note that the num. of bins (#bins) can vary across trials,
        reflecting the trial durations in the given `spiketrains` data.
    gpfa_instance : GPFA
        Instance of the GPFA() class in elephant, which was used to obtain
        `returned_data`.
    dimensions : 'all' or int or list of int, optional
        Dimensions to plot.
        Default: 'all'
    orthonormalized_dimensions : bool, optional
        Boolean which specifies whether to plot the orthonormalized latent
        state space dimension corresponding to the entry 'latent_variable_orth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'latent_variable' (False).
        Beware that the unconstrained state space dimensions 'latent_variable'
        are not ordered by their explained variance. These dimensions each
        represent one Gaussian process timescale $\tau$.
        On the contrary, the orthonormalized dimensions 'latent_variable_orth'
        are ordered by decreasing explained variance, allowing a similar
        intuitive interpretation to the dimensions obtained in a PCA. Due to
        the orthonormalization, these dimensions reflect mixtures of
        timescales.
        Default: True
    n_trials_to_plot : int, optional
        Number of single trial trajectories to plot.
        Default: 20
    trial_grouping_dict : dict or None
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the group name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
        Default: None
    colors : str or list of str, optional
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict. In case a string is given, all trials will
        share the same color unless `trial_grouping_dict` is specified, in
        which case colors will be set automatically to correspond to individual
        groups.
        Default: 'grey'
    plot_single_trajectories : bool, optional
        If True, single trial trajectories are plotted.
        Default: True
    plot_group_averages : bool, optional
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
        Default: False
    n_columns : int, optional
        Number of columns of the grid onto which the single plots are placed.
        The number of rows are deduced from the number of dimensions
        to be plotted.
        Default: 2
    plot_args_single : dict, optional
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_average : dict, optional
        Arguments dictionary passed to ax.plot() of the average trajectories.
    figure_args : dict, optional
        Arguments dictionary passed to matplotlib.pyplot.figure(),
        if ax is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    independent Poisson spike trains recorded in 50 trials with randomized
    rates up to 100 Hz and plot the resulting orthonormalized latent state
    space dimensions.

    .. plot::
        :include-source:

        import numpy as np
        import quantities as pq
        from elephant.gpfa import GPFA
        from elephant.spike_train_generation import homogeneous_poisson_process
        from viziphant.gpfa import plot_dimensions_vs_time
        np.random.seed(24)
        n_trials = 10
        n_channels = 5

        data = []
        for trial in range(n_trials):
            firing_rates = np.random.randint(low=1, high=100,
                                             size=n_channels) * pq.Hz
            spike_times = [homogeneous_poisson_process(rate=rate)
                           for rate in firing_rates]
            data.append(spike_times)
        gpfa = GPFA(bin_size=20 * pq.ms, x_dim=3, verbose=False)
        gpfa.fit(data)
        results = gpfa.transform(data, returned_data=['latent_variable_orth',
                                                      'latent_variable'])

        plot_dimensions_vs_time(
            returned_data=results,
            gpfa_instance=gpfa,
            dimensions=[0, 2],
            orthonormalized_dimensions=True,
            n_columns=1)
        plt.show()

    """
    if dimensions == 'all':
        dimensions = list(range(gpfa_instance.x_dim))
    elif isinstance(dimensions, int):
        dimensions = [dimensions]

    n_columns = min(n_columns, len(dimensions))

    # deduce n_rows from n_columns
    n_rows = math.ceil(len(dimensions) / n_columns)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns,
                             sharex=True, sharey=True, **figure_args)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        # (1, n) -> (n, 1)
        axes = axes.T

    data = _check_input_data(returned_data, orthonormalized_dimensions)
    if trial_grouping_dict is None:
        trial_grouping_dict = {}
    colors = _check_colors(colors, trial_grouping_dict, n_trials=data.shape[0])
    # infer n_trial from shape of the data
    n_trials = data.shape[0]
    # infer n_time_bins from maximal number of bins
    # TODO: deal with varying number of bins over trials?
    n_time_bins = gpfa_instance.transform_info['num_bins'].max()

    times = np.arange(1, n_time_bins + 1) * gpfa_instance.bin_size

    for dimension_index, axis in zip(dimensions, np.ravel(axes)):
        if plot_single_trajectories:
            for trial_idx in range(min(n_trials, n_trials_to_plot)):
                dat = data[trial_idx]

                key_id, trial_type = _get_trial_type(trial_grouping_dict,
                                                     trial_idx)

                # plot single trial trajectories
                axis.plot(times.magnitude,
                          dat[dimension_index, :],
                          color=colors[key_id],
                          label=trial_type,
                          **plot_args_single)

        if plot_group_averages:
            for color, trial_type in zip(colors, trial_grouping_dict.keys()):
                group_average = data[trial_grouping_dict[trial_type]].mean()
                axis.plot(times.magnitude,
                          group_average[dimension_index],
                          color=color,
                          label=trial_type,
                          **plot_args_average)

        _set_title_dimensions_vs_time(
            ax=axis,
            latent_variable_idx=dimension_index,
            orthonormalized_dimensions=orthonormalized_dimensions,
            data=data,
            gpfa_instance=gpfa_instance)

    _show_unique_legend(axes=axes[0, 0])
    plt.tight_layout()

    for axis in axes[-1, :]:
        axis.set_xlabel(f'Time ({times.dimensionality})')

    return fig, axes


def plot_trajectories(returned_data,
                      gpfa_instance,
                      dimensions=[0, 1],
                      block_with_cut_trials=None,
                      neo_event_name=None,
                      relevant_events=None,
                      orthonormalized_dimensions=True,
                      n_trials_to_plot=20,
                      trial_grouping_dict=None,
                      colors='grey',
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
                                              'label': 'start'}):

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

            `latent_variable_orth`: (#latent_vars, #bins) np.ndarray

            `latent_variable`:  (#latent_vars, #bins) np.ndarray

            `y`:  (#units, #bins) np.ndarray

            `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

            `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

        Note that the num. of bins (#bins) can vary across trials,
        reflecting the trial durations in the given `spiketrains` data.
    gpfa_instance : GPFA
        Instance of the GPFA() class in elephant, which was used to obtain
        returned_data.
    dimensions : list of int, optional
        List specifying the indices of the dimensions to use for the
        2D or 3D plot.
        Default: [0, 1]
    block_with_cut_trials : neo.Block or None, optional
        The neo.Block should contain each single trial as a separate
        neo.Segment including the neo.Event with a specified
        `neo_event_name`.
        Default: None
    neo_event_name : str or None, optional
        A string specifying the name of the neo.Event which should be used
        to identify the event times and labels of the `relevant_events`.
        Default: None
    relevant_events : list of str or None, optional
        List of names of the event labels that should be plotted onto each
        single trial trajectory.
        Default: None
    orthonormalized_dimensions : bool, optional
        Boolean which specifies whether to plot the orthonormalized latent
        state space dimension corresponding to the entry 'latent_variable_orth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'latent_variable' (False).
        Beware that the unconstrained state space dimensions 'latent_variable'
        are not ordered by their explained variance. These dimensions each
        represent one Gaussian process timescale $\tau$.
        On the contrary, the orthonormalized dimensions 'latent_variable_orth'
        are ordered by decreasing explained variance, allowing a similar
        intuitive interpretation to the dimensions obtained in a PCA. Due to
        the orthonormalization, these dimensions reflect mixtures of
        timescales.
        Default: True
    n_trials_to_plot : int, optional
        Number of single trial trajectories to plot.
        Default: 20
    trial_grouping_dict : dict or None, optional
        Dictionary which specifies the groups of trials which belong together
        (e.g. due to same trial type). Each item specifies one group: its
        key defines the group name (which appears in the legend) and the
        corresponding value is a list or np.ndarray of trial IDs.
        Default: None
    colors : str or list of str, optional
        List of strings specifying the colors of the different trial groups.
        The length of this list should correspond to the number of items
        in trial_grouping_dict. In case a string is given, all trials will
        share the same color unless `trial_grouping_dict` is specified, in
        which case colors will be set automatically to correspond to individual
        groups.
        Default: 'grey'
    plot_single_trajectories : bool, optional
        If True, single trial trajectories are plotted.
        Default: True
    plot_group_averages : bool, optional
        If True, trajectories of those trials belonging together specified
        in the trial_grouping_dict are averaged and plotted.
        Default: False
    plot_args_single : dict, optional
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_marker : dict, optional
        Arguments dictionary passed to ax.plot() for the single trial events.
    plot_args_average : dict, optional
        Arguments dictionary passed to ax.plot() of the average trajectories.
        if ax is None.
    plot_args_marker_start : dict, optional
        Arguments dictionary passed to ax.plot() for the marker of the
        average trajectory start.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    independent Poisson spike trains recorded in 50 trials with randomized
    rates up to 100 Hz and plot the resulting orthonormalized latent state
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
    >>> results = gpfa.transform(data, returned_data=['latent_variable_orth',
    ...                                               'latent_variable'])

    >>> trial_id_lists = np.arange(50).reshape(5,10)
    >>> trial_group_names = ['A', 'B', 'C', 'D', 'E']
    >>> trial_grouping_dict = {}
    >>> for trial_group_name, trial_id_list in zip(trial_group_names,
    ...                                            trial_id_lists):
    >>>     trial_grouping_dict[trial_group_name] = trial_id_list
    ...
    >>> plot_trajectories(
    ...        results,
    ...        gpfa,
    ...        block_with_cut_trials=None,
    ...        relevant_events=None,
    ...        dimensions=[0,1,2],
    ...        trial_grouping_dict=trial_grouping_dict,
    ...        plot_group_averages=False,
    ...        plot_single_trajectories=True,
    ...        n_trials_to_plot=200,
    ...        plot_args_single={'linewidth': 0.8,
    ...                          'alpha': 0.4, 'linestyle': '-'})

    """
    # prepare the input
    projection, n_dimensions = _check_dimensions(gpfa_instance, dimensions)
    data = _check_input_data(returned_data, orthonormalized_dimensions)
    if trial_grouping_dict is None:
        trial_grouping_dict = {}
    colors = _check_colors(colors, trial_grouping_dict, n_trials=data.shape[0])

    # infer n_trial from shape of the data
    n_trials = data.shape[0]

    # initialize figure and axis
    fig = plt.figure()
    axes = fig.gca(projection=projection, aspect='auto')

    # loop over trials
    if plot_single_trajectories:
        for trial_idx in range(min(n_trials, n_trials_to_plot)):
            dat = data[trial_idx][dimensions, :]
            key_id, trial_type = _get_trial_type(trial_grouping_dict,
                                                 trial_idx)
            color = colors[key_id]
            axes.plot(*dat,
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
                    dat_event = [[dat[i][event_time]]
                                 for i in range(n_dimensions)]
                    axes.plot(*dat_event,
                              marker=next(marker),
                              label=event_label,
                              color=color,
                              **plot_args_marker)

    if plot_group_averages:
        for color, trial_type in zip(colors, trial_grouping_dict.keys()):
            group_average = data[trial_grouping_dict[trial_type]].mean()
            group_average = group_average[dimensions, :]

            axes.plot(*group_average,
                      color=color,
                      label=trial_type,
                      **plot_args_average)
            axes.plot(*group_average[:, 0],
                      color=color,
                      **plot_args_marker_start)

    _set_axis_labels_trajectories(axes,
                                  orthonormalized_dimensions,
                                  dimensions)

    _show_unique_legend(axes=axes)
    plt.tight_layout()

    return fig, axes


def _check_input_data(returned_data, orthonormalized_dimensions):
    # by default returned_data is an array containing the
    # orthonormalized posterior mean of latent variable
    if isinstance(returned_data, np.ndarray):
        return returned_data
    if isinstance(returned_data, dict):
        if orthonormalized_dimensions:
            return returned_data['latent_variable_orth']
        if 'latent_variable' in returned_data.keys():
            return returned_data['latent_variable']
    raise ValueError("The latent variables before "
                     "orthonormalization 'latent_variable' are not in the "
                     "returned data")


def _check_colors(colors, trial_grouping_dict, n_trials):
    if trial_grouping_dict:
        if isinstance(colors, str) or len(colors) != len(trial_grouping_dict):
            colors = [f'C{i}' for i in range(len(trial_grouping_dict))]
    elif isinstance(colors, str):
        colors = [colors] * n_trials
    return colors


def _check_dimensions(gpfa_instance, dimensions):
    n_dimensions = len(dimensions)
    if gpfa_instance.x_dim < n_dimensions:
        raise ValueError(f"GPFA trajectories dimensionality "
                         f"({gpfa_instance.x_dim}) is lower than the "
                         f"requested ({n_dimensions})")
    if n_dimensions not in (2, 3):
        raise ValueError("Pick only 2 or 3 dimensions to visualize.")
    projection = None if n_dimensions == 2 else '3d'
    return projection, n_dimensions


def _get_trial_type(trial_grouping_dict, trial_idx):
    for key_id, (trial_type, trial_ids) in enumerate(
            trial_grouping_dict.items()):
        if trial_idx in trial_ids:
            return key_id, trial_type
    return 0, None


def _set_title_dimensions_vs_time(ax,
                                  latent_variable_idx,
                                  orthonormalized_dimensions,
                                  data,
                                  gpfa_instance):
    if orthonormalized_dimensions:
        title = r'$\tilde{{\mathbf{{x}}}}_{{{},:}}$'.format(
            latent_variable_idx)

        # percentage of variance of the dimensionality reduced data
        # that is explained by this latent variable
        variances = [np.var(np.hstack(data)[i, :]) for
                     i in range(gpfa_instance.x_dim)]
        total_variance = np.sum(variances)
        explained_variance = np.round(
            variances[latent_variable_idx]/total_variance*100, 2)

        title = title + f'% exp. var.: {explained_variance} %'
    else:
        title = r"${{\mathbf{{x}}}}_{{{},:}}$".format(latent_variable_idx)

        # time scale of the gaussian process associated to this latent variable
        gamma = gpfa_instance.params_estimated['gamma'][latent_variable_idx]
        GP_time_scale = np.round(gpfa_instance.bin_size / np.sqrt(gamma), 2)

        title = title + rf'$\tau$: {GP_time_scale}'

    ax.set_title(title, fontsize=16)


def _set_axis_labels_trajectories(ax,
                                  orthonormalized_dimensions,
                                  dimensions):
    if orthonormalized_dimensions:
        str1 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[0])
        str2 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[1])
        if len(dimensions) == 3:
            str3 = r"$\tilde{{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[2])
    else:
        str1 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[0])
        str2 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[1])
        if len(dimensions) == 3:
            str3 = r"${{\mathbf{{x}}}}_{{{},:}}$".format(dimensions[2])
    ax.set_xlabel(str1, fontsize=16)
    ax.set_ylabel(str2, fontsize=16)
    if len(dimensions) == 3:
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


def _show_unique_legend(axes):
    # only plot unique labels
    handles, labels = axes.get_legend_handles_labels()
    if len(handles) == 0:
        # no labels have been provided
        return
    by_label = dict(zip(labels, handles))
    axes.legend(by_label.values(), by_label.keys())
