from .Glm import Glm
from sklearn import linear_model

model = linear_model.LogisticRegression(verbose = 0, max_iter = 1000000000, n_jobs = -1)

class LogReg_glm(Glm):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, model)

    @staticmethod
    def hello_world():
        print('Log Regression GLM Model')

    @staticmethod
    def get_settings():
        return model.get_params()
