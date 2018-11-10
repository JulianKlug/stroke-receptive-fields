import os
import numpy as np
from torch import nn
from vxl_NN.Torch_model import Torch_model

EPOCHS = 100

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class LogisticRegression(nn.Sequential):
    def __init__(self, n_features, n_channels_out):
        super(LogisticRegression, self).__init__()
        self.l0 = Flatten()
        self.l1 = nn.Linear(n_features, n_channels_out)

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        return x

    @staticmethod
    def get_params():
        return {}


# class LogisticRegression(nn.Sequential):
#     def __init__(self, n_channels, n_channels_out, rf):
#         super(LogisticRegression, self).__init__()
#         self.l1 = nn.Conv3d(n_channels, n_channels_out, 2 * np.max(rf) + 1)
#         self.softmax = nn.Softmax(dim = 0)
#
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.softmax(x)
#         return x
#
#     @staticmethod
#     def get_params():
#         return {}

class LogReg_NN(Torch_model):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        n_features = n_channels * ((2 * np.max(rf)) + 1)**3
        print(n_features)
        super().__init__(fold_dir, fold_name, LogisticRegression(n_features, n_channels_out), n_channels = n_channels, rf_dim = rf, n_epochs = EPOCHS)

    @staticmethod
    def hello_world():
        print('Log Regression NN Model')

    @staticmethod
    def get_settings():
        return LogisticRegression.get_params()
