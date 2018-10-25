import os
import numpy as np
from vxl_glm.Glm import Glm
from sklearn import linear_model

model = linear_model.LogisticRegression(verbose = 0, max_iter = 1000000000, n_jobs = -1)

class LogReg_glm(Glm):
    def __init__(self, fold_dir, fold_name):
        super().__init__(fold_dir, fold_name, model)

    @staticmethod
    def hello_world():
        print('Log Regression GLM Model')

    @staticmethod
    def get_settings():
        return model.get_params()
