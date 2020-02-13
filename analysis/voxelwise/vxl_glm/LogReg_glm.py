from .Glm import Glm
from sklearn import linear_model

base_model = linear_model.LogisticRegression(verbose = 0, max_iter = 1000000000, n_jobs = -1)

class LogReg_glm(Glm):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1, model=None):
        if model is None:
            super().__init__(fold_dir, fold_name, base_model)
        else:
            print('Loading from saved model')
            super().__init__(fold_dir, fold_name, model, pre_trained=True)


    @staticmethod
    def hello_world():
        print('Log Regression GLM Model')

    @staticmethod
    def get_settings():
        return base_model.get_params()
