"""Contains losses defined as per DeepVesselNet paper by Giles Tetteh"""

import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, targets):
        _, pred = torch.max(outputs, 1)
        
        pred = pred.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        numerator = 2. * torch.sum(pred * targets)
        denominator = torch.sum(pred + targets)
        
        # loss = Variable((1. - (numerator + 1)/(denominator + 1)), requires_grad=True)
        loss = Variable((1. - (numerator/denominator)), requires_grad=True)

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
            print(c_true.shape)
            print(L.shape)
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
