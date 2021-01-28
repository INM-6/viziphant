"""
Tensor Component Analysis (TCA) plots
---------------------------------------------

Visualizes output from `elephant.tensor_component_analysis`.

.. autosummary::
    :toctree: toctree/tensor_component_analysis/

"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from pathlib import Path


# TODO this is very V4A specific
# make generic and return figure and axes to customize if needed

def plot_factor(factors,
                dim,
                trial_ids=None,
                unit_ids=None,
                unit_names=None,
                LFP_chans=None,
                block=None,
                targdict_list=None,
                sort_trial_factor=False,
                figure_destination='./Figures'):
    """
    For a given dimension, plot the neural factors, trial factors, and time
    factors.
    trial_ids is list/array of length trials with the type of each trial
        ex: ['PGHF', 'SGHF', ...]
    unit_ids: list of unique unit ids
        ex: [2.1, 2.2, 5.1, ...]
    """
    plt.figure(figsize=(15, 15))
    nHist = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=2)
    nMap = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
    trial = plt.subplot2grid((3, 3), (1, 0), colspan=3)
    time = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    if type(factors[0]) != np.ndarray:
        for i, f in enumerate(factors):
            factors[i] = f.numpy()

    n_units = factors[0].shape[0]
    n_trials = factors[1].shape[0]
    n_time_bins = factors[2].shape[0]

    if targdict_list:
        sua_mua_list = []
        snr_list = []
        for st in targdict_list:
            sua_mua_list.append(st.annotations['unit_type'])
            snr_list.append(st.annotations['snr'])

        sua_mask = np.array(sua_mua_list) == 'sua'
        unit_colors = ['black' if i else 'grey' for i in sua_mask]
    else:
        unit_colors = ['black' for i in range(n_units)]

    nHist.set_title(f"Neural factor {dim}")
    nHist.bar(np.arange(0, n_units),
              factors[0][:, dim],
              color=unit_colors)
    nHist.set_xlabel("Neuron")
    if unit_names is not None:
        nHist.set_xticks(np.arange(0, n_units))
        nHist.set_xticklabels(unit_names, rotation=90)

    nMap.set_title("Mean unit weight per channel")
    # if (unit_ids is not None) and (LFP_chans is None):
    #     w = transform_weights_to_utah_map(factors[0][:, dim], unit_ids)
    #     im = nMap.imshow(w, aspect='auto', cmap='Reds', vmin=0, origin='lower')
    #     plt.colorbar(im, ax=nMap)
    # elif (unit_ids is None) and (LFP_chans is not None):
    #     w = ut.transform_true_channels_to_utah_map(factors[0][:, dim], LFP_chans)
    #     vmin = np.min(w)
    #     vmax = np.max(w)
    #     lim = np.max((abs([vmin, vmax])))
    #     im = nMap.imshow(w, aspect='auto', cmap='seismic', vmin=-lim, vmax=lim)
    #     plt.colorbar(im, ax=nMap)

    # TODO enable some annotations like this in a generic way
    previously_unsuccessful_list = []
    for seg in block.segments:
        previously_unsuccessful_list.append(
            seg.annotations['previous_unsuccessful_trials'])

    trial.set_title(f"Trial factors {dim}")
    cmap_name = 'tab20'
    cmap = get_cmap(cmap_name)
    colors = cmap.colors
    # trial.set_prop_cycle(color=colors)
    markers = Line2D.filled_markers

    if sort_trial_factor:
        if trial_ids is not None:
            trial_ids = np.array(trial_ids)
            sort_indices = np.argsort(trial_ids)
            for id in np.unique(trial_ids[sort_indices]):
                mask = (np.array(trial_ids[sort_indices]) == id)
                ran = np.argwhere(np.array(trial_ids[sort_indices]) == id)
                color = colors[id-1]
                marker = markers[id]
                trial.plot(ran, factors[1][sort_indices, :][mask, dim],
                           marker=marker,
                           linestyle='None',
                           label=id,
                           color=color)
                mean = np.mean(factors[1][sort_indices, :][mask, dim])
                std = np.std(factors[1][sort_indices, :][mask, dim])
                trial.axhline(mean, color=color)
                trial.vlines(id, ymin=mean-std, ymax=mean+std, color=color)

            trial.legend()

        else:
            trial.plot(factors[1][:, dim], 'o', color='k')
        trial.set_xlabel("Trial")

        for id, n_unsuccessful in enumerate(np.array(previously_unsuccessful_list)[sort_indices]):
            if n_unsuccessful > 0:
                trial.plot(id, factors[1][sort_indices, :]
                           [id, dim], 'x', color='red', markersize=10)
                trial.text(id, factors[1][sort_indices, :][id, dim], str(n_unsuccessful),
                           horizontalalignment='left',
                           verticalalignment='bottom',)

    else:
        if trial_ids is not None:
            for id in np.unique(trial_ids):
                mask = (np.array(trial_ids) == id)
                ran = np.argwhere(np.array(trial_ids) == id)
                color = colors[id-1]
                marker = markers[id]
                trial.plot(ran, factors[1][mask, dim],
                           marker=marker,
                           linestyle='None',
                           label=id,
                           color=color)
                mean = np.mean(factors[1][mask, dim])
                std = np.std(factors[1][mask, dim])
                trial.axhline(mean, color=color)
                trial.vlines(id, ymin=mean-std, ymax=mean+std, color=color)

            trial.legend()

        else:
            trial.plot(factors[1][:, dim], 'o', color='k')
        trial.set_xlabel("Trial")

        for id, n_unsuccessful in enumerate(previously_unsuccessful_list):
            if n_unsuccessful > 0:
                trial.plot(id, factors[1][id, dim], 'x',
                           color='red', markersize=10)
                trial.text(id, factors[1][id, dim], str(n_unsuccessful),
                           horizontalalignment='left',
                           verticalalignment='bottom',)

    time.set_title(f"Time factor {dim}")
    t_start = block.segments[0].t_start
    t_stop = block.segments[0].t_stop
    time_axis = np.linspace(t_start, t_stop, n_time_bins)
    time.plot(time_axis, factors[2][:, dim], color='k')
    time.set_xlabel(f'(warped) time [{time_axis.units}]')

    # TODO leave this type of customization to the user
    representative_segment = block.segments[0]
    warped_event_labels = representative_segment.annotations[
        'warped_event_labels']
    new_event_times = representative_segment.annotations['new_event_times']
    previous_time = 0
    for t, label, color in zip(
        new_event_times,
        warped_event_labels,
        colors[:len(new_event_times)]
    ):
        # TODO implement mechanism to avoid event labels to overlap
        if t != previous_time:
            time.axvline(t, color=color, zorder=0)
            time.text(t, time.get_ylim()[1], label,
                      color=color,
                      horizontalalignment='left',
                      verticalalignment='bottom',
                      rotation=40)
        previous_time = t

    plt.tight_layout()

    Path(figure_destination).mkdir(parents=False, exist_ok=True)

    plt.savefig(f'{figure_destination}/all_components_'+str(dim)+'.png',
                facecolor='white', edgecolor='none', bbox_inches='tight')
