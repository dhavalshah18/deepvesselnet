"""Defines the class DeepVesselNetFCN."""

import torch
import torch.nn as nn
import math


class DeepVesselNetFCN(nn.Module):
    """INPUT - 3DCONV - 3DCONV - 3DCONV - 3DCONV - FCN """
    def __init__(self, nchannels=1, nlabels=2, dim=3, batchnorm=True, dropout=False):
        """
        Builds the network structure with the provided parameters

        Input:
        - nchannels (int): number of input channels to the network
        - nlabels (int): number of labels to be predicted
        - dim (int): dimension of the network
        - batchnorm (boolean): sets if network should have batchnorm layers
        - dropout (boolean): set if network should have dropout layers
        """
        super().__init__()

        self.nchannels = nchannels
        self.nlabels = nlabels
        self.dims = dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=self.nchannels, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=5, out_channels=10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(in_channels=10, out_channels=20, kernel_size=5, padding=2)
        self.conv4 = nn.Conv3d(in_channels=20, out_channels=50, kernel_size=3, padding=1)
        
        # Fully Convolutional layer
        self.fcn1 = nn.Conv3d(in_channels=50, out_channels=self.nlabels, kernel_size=1)
        
        # Batch Normalisation
        self.batchnorm1 = nn.BatchNorm3d(5)
        self.batchnorm2 = nn.BatchNorm3d(10)
        self.batchnorm3 = nn.BatchNorm3d(20)
        self.batchnorm4 = nn.BatchNorm3d(50)
        
        # Dropout
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        
        # Softmax layer
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
        
        for param in self.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        
        # 1st layer
        x = self.conv1(x)
        if self.batchnorm:
            x = self.batchnorm1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout1(x)
        
        # 2nd layer
        x = self.conv2(x)
        if self.batchnorm:
            x = self.batchnorm2(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout2(x)
        
        # 3rd layer
        x = self.conv3(x)
        if self.batchnorm:
            x = self.batchnorm3(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout3(x)
        
        x = self.conv4(x)
#         x = self.batchnorm4(x)
        x = self.relu(x)
#         x = self.dropout4(x)
        
        x = self.fcn1(x)
        x = self.sigmoid(x)
        
        x = self.softmax(x)
        return x
    
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)