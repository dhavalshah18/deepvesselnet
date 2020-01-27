import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.lines import Line2D
from torch.autograd import Variable
from itertools import product


def dice_coeff(outputs, targets, smooth=1, pred=False):
    if pred:
        pred = outputs
    else:
        _, pred = torch.max(outputs, 1)

    pred = F.one_hot(pred.long(), num_classes=2)
    targets = F.one_hot(targets.long(), num_classes=2)
    
    dim = tuple(range(1, len(pred.shape)-1))
    intersection = torch.sum(targets * pred, dim=dim, dtype=torch.float)
    union = torch.sum(targets, dim=dim, dtype=torch.float) + torch.sum(pred, dim=dim, dtype=torch.float)
    
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), dtype=torch.float)
        
    return dice


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            print(p.grad)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


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
        divisor[:, (i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w, (k * s_d):(k * s_d) + p_d] += 1
    volume /= divisor
    return volume[:, 0:r_h, 0:r_w, 0:r_d]

def test(model, volume, patch_size=64, stride=60):
    model.eval()

    patch_size = (1, patch_size, patch_size, patch_size)
    stride = (stride, stride, stride)

    patches = patchify(volume, patch_size, stride)
    patch_shape = patches.shape
    patches = patches.view((-1,) + patch_size).cuda().type(torch.cuda.FloatTensor)

    output = torch.zeros((0, ) + patch_size[1:]).type(torch.FloatTensor)

    batch_size = 5 # user input
    num = int(np.ceil(1.0 * patches.shape[0] / batch_size))

    for i in range(num):
        model_output = model.forward(patches[batch_size*i:batch_size*i + batch_size])

        _, preds = torch.max(model_output, 1)
        preds = preds.cuda().type(torch.FloatTensor)

        output = torch.cat((output, preds), 0)

    new_shape = patch_shape
    output = unpatchify(output.view(new_shape), stride, volume.shape, 1)
    output = output.squeeze(0)
    
    return output
    
    