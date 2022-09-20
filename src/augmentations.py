import random
import numpy as np


def random_crop(lr: np.ndarray, hr: np.ndarray, scale: int = 2, patch_size: int = 64) -> tuple:
    """
    Extracts a random patch of the given size from an input lr image and the corresponding hr patch scaled using the
    given scale.

    :param lr: low-resolution image to extract the patch from (ndarray)
    :param hr: high-resolution image to extract the patch from (ndarray)
    :param scale: scale to use for the extraction of the hr patch (int, default 2)
    :param patch_size: size of the low resolution (square) patch (int, default 64)
    :return: tuple containing the extacted lr and hr patches
    """

    # extract size of the lr image
    height, width = lr.shape[:-1]

    # extract random starting coordinates of the patch in the lr image
    x = random.randint(0, width - patch_size)
    y = random.randint(0, height - patch_size)

    # compute the starting coordinates of the patch in the hr image
    hr_patch_size = patch_size * scale
    hx, hy = x * scale, y * scale

    # extract the patch from the two images
    lr = lr[y:y + patch_size, x:x + patch_size]
    hr = hr[hy:hy + hr_patch_size, hx:hx + hr_patch_size]

    return lr, hr


def random_horizontal_flip(lr: np.ndarray, hr: np.ndarray, p: float = .5) -> tuple:
    """
    Randomly applies horizontal flip to the given lr and hr patches with the given flipping probability

    :param lr: low-resolution patch to flip (ndarray)
    :param hr: high-resolution patch to flip (ndarray)
    :param p: probability of the flipping (float, default 0.5)
    :return: tuple containing the flipped (or not) lr and hr patches
    """

    # flip horizontally the images
    if random.random() < p:
        lr = np.fliplr(lr)
        hr = np.fliplr(hr)

    return lr, hr


def random_90_rotation(lr: np.ndarray, hr: np.ndarray) -> tuple:
    """
    Randomly applies a 90° rotation (or not) to the given lr and hr patches

    :param lr: low-resolution patch to rotate (ndarray)
    :param hr: high-resolution patch to rotate (ndarray)
    :return: tuple containing the rotated (or not) lr and hr patches
    """

    # choose a rotation angle (0, 90, -90)
    n_rotations = random.choice([0, 1, 3])

    # rotate the images
    lr = np.rot90(lr, n_rotations)
    hr = np.rot90(hr, n_rotations)

    return lr, hr
