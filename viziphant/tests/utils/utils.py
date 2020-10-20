import hashlib
import io
import os
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

TESTS_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TESTS_DIR / "data"
TARGET_IMAGES_DIR = TESTS_DIR / "target_images"

TARGET_IMAGES_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        raise IOError(f"Invalid file: '{fpath}' does not exist.")
    is_valid = md5 == calculate_md5(fpath)
    if not is_valid:
        raise IOError("Invalid file: wrong checksum.")


def images_difference(path_target_image, path_result_image):
    """
    Computes normalized images difference.

    Parameters
    ----------
    path_target_image : str or io.BytesIO
        The file-like target image.
    path_result_image : str or io.BytesIO
        The file-like result image to compare with the target.

    Returns
    -------
    diff_norm : float
        The L1-norm of the difference between two input images per pixel per
        channel.
    """
    # imread returns RGBA image
    target_image = mpimg.imread(path_target_image, format='png')
    result_image = mpimg.imread(path_result_image, format='png')
    if result_image.shape != target_image.shape:
        raise ValueError("Images have different shapes")
    diff_image = np.abs(target_image - result_image)
    diff_norm = np.linalg.norm(diff_image.flatten(), ord=1)
    diff_norm = diff_norm / diff_image.size  # per pixel per channel
    return diff_norm
