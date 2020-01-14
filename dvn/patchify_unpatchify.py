import numpy as np
import torch
from torch.nn import functional as F
from itertools import product

def patchify(volume, patch_size, step):
    """

    :param volume:
    :param patch_size:
    :param step:
    :return:
    """
    assert len(volume.shape) == 4

    _, v_h, v_w, v_d = volume.shape

    s_h, s_w, s_d = step

    _, p_h, p_w, p_d = patch_size

    # Calculate the number of patch in each axis
    n_w = np.ceil(1.0*(v_w-p_w)/s_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/s_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/s_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_w = (n_w - 1) * s_w + p_w - v_w
    pad_h = (n_h - 1) * s_h + p_h - v_h
    pad_d = (n_d - 1) * s_d + p_d - v_d
    # print(volume.shape, (0, pad_h, 0, pad_w, 0, pad_d))
    volume = F.pad(volume, (0, pad_d, 0, pad_w, 0, pad_h), 'constant')
    # print(volume.shape)
    patches = torch.zeros((n_h, n_w, n_d,)+patch_size, dtype=volume.dtype)

    for i, j, k in product(range(n_h), range(n_w), range(n_d)):
        patches[i, j, k] = volume[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d]

    return patches


def unpatchify(patches, step, imsize, scale_factor):
    """

    :param patches:
    :param step:
    :param imsize:
    :param scale_factor:
    :return:
    """
    assert len(patches.shape) == 7

    c, r_h, r_w, r_d = imsize
    s_h, s_w, s_d = tuple(scale_factor*np.array(step))

    n_h, n_w, n_d, _, p_h, p_w, p_d = patches.shape

    v_w = (n_w - 1) * s_w + p_w
    v_h = (n_h - 1) * s_h + p_h
    v_d = (n_d - 1) * s_d + p_d

    volume = torch.zeros((c, v_h, v_w, v_d), dtype=patches.dtype)
    divisor = torch.zeros((c, v_h, v_w, v_d), dtype=patches.dtype)
#     print(volume.shape, imsize)

    for i, j, k in product(range(n_h), range(n_w), range(n_d)):
        patch = patches[i, j, k]
        volume[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d] += patch
        divisor[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d] += 1.
    volume /= divisor
    return volume[:, 0:r_h, 0:r_w, 0:r_d]