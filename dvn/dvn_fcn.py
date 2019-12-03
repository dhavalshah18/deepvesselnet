"""Defines the class DeepVesselNetFCN."""

import torch
import torch.nn as nn


class DeepVesselNetFCN(nn.Module):
    """INPUT - 3DCONV - 3DCONV - 3DCONV - 3DCONV - FCN """
    def __init__(self, nchannels=1, nlabels=2, cross_hair=False, dim=3):
        """
        Builds the network structure with the provided parameters

        Input:
        - nchannels (int): number of input channels to the network
        - nlabels (int): number of labels to be predicted
        - cross_hair (boolean): whether to use cross hair filters or classical convolution filters
        - dim (int): dimension of the network
        """
        super(DeepVesselNetFCN, self).__init__()

        self.nchannels = nchannels
        self.nlabels = nlabels

        self.classifier = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=3, padding_mode='same'),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=5, out_channels=10, kernel_size=5, padding_mode='same'),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=10, out_channels=20, kernel_size=5, padding_mode='same'),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=20, out_channels=50, kernel_size=3, padding_mode='same'),
            nn.ReLU(inplace=True),

            # This makes sense, but not sure if the Keras dvn is the same
            nn.Linear(in_features=50, out_features=self.nlabels),
            nn.Sigmoid()

        )

        # Not sure what dim is here
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):

        x = self.classifier(x)
        # WHAT DO I DO WITH THE SOFTMAX??
        x = self.softmax(x)
        return x
