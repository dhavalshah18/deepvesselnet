import numpy as np
import torch

def to_one_hot(data, cls=None):
    if cls is None:
        cls = len(np.unique(data))
    sh = data.shape
    hot = np.zeros((np.prod(sh), cls), dtype=data.dtype)
    hot[np.arange(np.prod(sh)), data.flatten()] = 1
    return np.reshape(hot, sh + (cls, ))

def dice_coeff(outputs, targets, smooth=1):
    _, pred = torch.max(outputs, 1)
    intersection = torch.sum(targets * pred, dim=[1, 2, 3])
    union = torch.sum(targets, dim=[1, 2, 3]) +  torch.sum(pred, dim=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice