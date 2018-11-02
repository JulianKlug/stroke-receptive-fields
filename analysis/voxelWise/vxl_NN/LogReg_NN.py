import os
import numpy as np
from torch import nn
from vxl_NN.Torch_model import Torch_model

class LogisticRegression(nn.Sequential):
    def __init__(self, n_channels, n_channels_out, rf):
        super(LogisticRegression, self).__init__()
        self.l1 = nn.Conv3d(4, 1, 1)

    def forward(self, x):
        x = self.l1(x)
        return x

    @staticmethod
    def get_params():
        return {}

class LogReg_NN(Torch_model):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, LogisticRegression(n_channels, n_channels_out, rf))

    @staticmethod
    def hello_world():
        print('Log Regression NN Model')

    @staticmethod
    def get_settings():
        return LogisticRegression.get_params()
