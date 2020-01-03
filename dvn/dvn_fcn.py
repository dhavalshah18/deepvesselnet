"""Defines the class DeepVesselNetFCN."""

import torch
import torch.nn as nn
import math


class DeepVesselNetFCN(nn.Module):
    """INPUT - 3DCONV - 3DCONV - 3DCONV - 3DCONV - FCN """
    def __init__(self, nchannels=1, nlabels=2, dim=3):
        """
        Builds the network structure with the provided parameters

        Input:
        - nchannels (int): number of input channels to the network
        - nlabels (int): number of labels to be predicted
        - dim (int): dimension of the network
        """
        super().__init__()

        self.nchannels = nchannels
        self.nlabels = nlabels
        self.dims = dim
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=self.nchannels, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=5, out_channels=10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(in_channels=10, out_channels=20, kernel_size=5, padding=2)
        self.conv4 = nn.Conv3d(in_channels=20, out_channels=50, kernel_size=3, padding=1)
        
        # Fully Convolutional layer
        self.fcn1 = nn.Conv3d(in_channels=50, out_channels=self.nlabels, kernel_size=1)
        
        # Softmax layer
        # DON'T KNOW WHAT DIM SHOULD BE
        self.softmax = nn.Softmax(dim=1)
        
        # Non-linearities
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Find upper and lower bound based on kernel size of layer
                lower = -1/math.sqrt(m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2])
                upper = 1/math.sqrt(m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2])
                # Uniformly initialize with upper and lower bounds
                m.weight = nn.init.uniform_(m.weight, a=lower, b=upper)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.sigmoid(self.fcn1(x))
        x = self.softmax(x)
        return x
