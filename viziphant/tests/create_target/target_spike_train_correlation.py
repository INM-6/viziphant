import matplotlib.pyplot as plt
import numpy as np
import seaborn
from quantities import s, Hz

import elephant.spike_train_correlation as stcorr
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_generation import homogeneous_poisson_process
from viziphant.spike_train_correlation import plot_corrcoef
from viziphant.tests.utils.utils import TARGET_IMAGES_DIR

CORRCOEF_TARGET_PATH = TARGET_IMAGES_DIR / "plot_correlation_coefficient.png"


def get_default_corrcoef_matrix():
    # set random seed explicitly, which is used in homogeneous_poisson_process,
    # to avoid using different seeds for creating target and result image
    np.random.seed(0)
    spike_train_1 = homogeneous_poisson_process(rate=10.0 * Hz,
                                                t_start=0.0 * s,
                                                t_stop=10.0 * s)
    spike_train_2 = homogeneous_poisson_process(rate=10.0 * Hz,
                                                t_start=0.0 * s,
                                                t_stop=10.0 * s)
    # the binsize of 0.1s is rather large so we might expect non-zero
    # cross-correlation
    corrcoef_matrix = stcorr.corrcoef(BinnedSpikeTrain(
        [spike_train_1, spike_train_2], binsize=0.1 * s))
    return corrcoef_matrix


def create_target_plot_correlation_coefficient():
    seaborn.set_style('ticks')
    target_image1_corrcoef, axes1 = plt.subplots(
        1, 1, subplot_kw={'aspect': 'equal'})

    plot_corrcoef(
        get_default_corrcoef_matrix(),
        axes1,
        correlation_minimum=-1.,
        correlation_maximum=1.,
        colormap='bwr', color_bar_aspect=20,
        color_bar_padding_fraction=.5)

    target_image1_corrcoef.savefig(CORRCOEF_TARGET_PATH)


if __name__ == '__main__':
    create_target_plot_correlation_coefficient()
