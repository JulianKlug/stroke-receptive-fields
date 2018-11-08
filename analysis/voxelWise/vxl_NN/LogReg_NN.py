import os
import numpy as np
from torch import nn
from vxl_NN.Torch_model import Torch_model

EPOCHS = 2

class LogisticRegression(nn.Sequential):
    def __init__(self, n_channels, n_channels_out, rf):
        super(LogisticRegression, self).__init__()
        self.l1 = nn.Conv3d(n_channels, n_channels_out, 2 * np.max(rf) + 1)
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        x = self.l1(x)
        x = self.softmax(x)
        return x

    @staticmethod
    def get_params():
        return {}

class LogReg_NN(Torch_model):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, LogisticRegression(n_channels, n_channels_out, rf), n_epochs = EPOCHS)

    @staticmethod
    def hello_world():
        print('Log Regression NN Model')

    @staticmethod
    def get_settings():
        return LogisticRegression.get_params()
