"""Contains losses defined as per DeepVesselNet paper by Giles Tetteh"""

import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np


def categorical_crossentropy():
    def loss(inputs, targets):
        L = nn.CrossEntropyLoss()
        return L(inputs, targets)
    return loss


def weighted_categorical_crossentropy(classes=2):
    def loss(inputs, targets):
        L1 = nn.CrossEntropyLoss(reduction='none')
        L = L1(inputs, targets)
        print(targets.shape)
        _, y_true_p = torch.max(targets)
        for c in range(classes):
            # element-wise equality
            c_true = torch.eq(y_true_p, c).type(inputs)
            w = 1. / (torch.sum(c_true))
            C = torch.sum(L * c_true * w) if c==0 else C+torch.sum(L * c_true *w)

        return C

    return loss


def weighted_categorical_crossentropy_fpr(classes=2, threshold=0.5):
    def loss(inputs, targets):
        L1 = nn.CrossEntropyLoss(reduction='none')
        L = L1(inputs, targets)
        _, y_true_p = torch.max(targets)
        y_pred_probs, y_pred_bin = torch.max(inputs)

        for c in range(classes):
            # element-wise equality
            c_true = torch.eq(y_true_p, c).type(inputs)
            w = 1. / (torch.sum(c_true))
            C = torch.sum(L * c_true * w) if c == 0 else C + torch.sum(L * c_true * w)

            # FP rate correction calculation
            # False prediction
            c_false_p = torch.ne(y_true_p, c) * torch.eq(y_pred_bin, c)
            # Calculate gamma
            gamma = 0.5 + (torch.sum(torch.abs((c_false_p * y_pred_probs) - 0.5)) / (torch.sum(c_false_p)))
            wc = w * gamma
            C = C + torch.sum(L * c_false_p * wc)

        return C

    return loss