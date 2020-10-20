import io
import unittest

import matplotlib.pyplot as plt
import seaborn

from viziphant.spike_train_correlation import plot_corrcoef
from viziphant.tests.create_target.target_spike_train_correlation \
    import CORRCOEF_TARGET_PATH, get_default_corrcoef_matrix, \
    create_target_plot_correlation_coefficient
from viziphant.tests.utils.utils import images_difference


class SpikeTrainCorrelationTestCase(unittest.TestCase):
    def test_corroef(self):
        # TODO: remove creating target image function after setting up \
        #  git-lfs
        create_target_plot_correlation_coefficient()

        seaborn.set_style('ticks')
        result_image_corrcoef, axes2 = plt.subplots(
            1, 1, subplot_kw={'aspect': 'equal'})

        plot_corrcoef(get_default_corrcoef_matrix(),
                      axes2,
                      correlation_minimum=-1.,
                      correlation_maximum=1.,
                      colormap='bwr', color_bar_aspect=20,
                      color_bar_padding_fraction=.5)

        with io.BytesIO() as buf:
            result_image_corrcoef.savefig(buf, format="png")
            buf.seek(0)
            diff_norm = images_difference(str(CORRCOEF_TARGET_PATH), buf)
        tolerance = 1e-3
        self.assertLessEqual(diff_norm, tolerance)


if __name__ == '__main__':
    unittest.main()
