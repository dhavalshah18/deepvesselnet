"""Contains losses defined as per DeepVesselNet paper by Giles Tetteh"""

import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from dvn import misc as ms


class Soft_Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, targets):
        _, pred = torch.max(outputs, 1)
        
        # Change to one hot encoding
        # Shape N x X x Y x Z x C (classes)
        pred = torch.as_tensor(ms.to_one_hot(pred.cpu().numpy(), cls=2))
        targets = torch.as_tensor(ms.to_one_hot(targets.cpu().numpy(), cls=2))
        
        dim = tuple(range(1, len(pred.shape)-1))
        numerator = 2. * torch.sum(targets * pred, dim=dim)
        denominator = torch.sum(targets**2 + pred**2, dim=dim)

        loss = Variable((1. - torch.mean(numerator/denominator)), requires_grad=True)

        return loss
        
        
# Doesn't work yet
# Need to fix forward
class Weighted_CCE(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes
        
    def forward(self, outputs, targets):
        loss = nn.CrossEntropyLoss(reduction='none')
        L = loss(outputs, targets)
        y_max, y_true_p = torch.max(targets, 1)
        
        for c in range(self.classes):
            c_true = torch.eq(y_true_p, c).type(outputs.dtype)
            w = 1. / (torch.sum(c_true))
            C = torch.sum(L*c_true*w) if c==0 else C + torch.sum(L*c_true*w)

        return C

# Doesn't work yet
# Need to fix forward
class Weighted_CCE_FPR(nn.Module):
    def __init__(self, classes=2, threshold=0.5):
        super().__init__()
        self.classes = classes
        self.threshold = threshold
        
    def forward(outputs, targets):
        L1 = nn.CrossEntropyLoss(reduction='none')
        L = L1(outputs, targets)
        y_max, y_true_p = torch.max(targets)
        y_outputs_probs, y_outputs_bin = torch.max(outputs)

        for c in range(classes):
            # element-wise equality
            c_true = torch.eq(y_true_p, c).type(outputs.dtype).unsqueeze(0)
            w = 1. / (torch.sum(c_true))
            C = torch.sum(L*w*c_true) if c == 0 else C + torch.sum(L*w*c_true)

            # FP rate correction calculation
            # False outputsiction
            c_false_p = torch.ne(y_true_p, c) * torch.eq(y_outputs_bin, c)
            # Calculate gamma
            gamma = 0.5 + (torch.sum(torch.abs((c_false_p * y_outputs_probs) - 0.5)) / (torch.sum(c_false_p)))
            wc = w * gamma
            C = C + torch.sum(L * c_false_p * wc)

        return C
