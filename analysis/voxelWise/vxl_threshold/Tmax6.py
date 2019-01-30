import numpy as np
from vxl_threshold.Treshold_Model import Treshold_Model

class Tmax6_treshold():
    def __init__(self):
        self.threshold = 6

    def fit(self, X_train, y_train):
        return self

    def predict_proba(self, data):
        output = np.zeros(data.shape)
        output[data > self.threshold] = 1
        return np.squeeze(output)

class Tmax6(Treshold_Model):
    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, model = Tmax6_treshold())

    @staticmethod
    def hello_world():
        print('Tmax > 6s threshold Model')
        print('CAN ONLY BE USED WITH TMAX')
        print('Feature scaling is not allowed, as it changes the threshold')

    @staticmethod
    def get_settings():
        return ""

def Tmax6_Model_Generator(X_shape, feature_scaling):
    """
    Model Generator for Tmax > 6s threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c), where c = 1
        feature_scaling: boolean

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the threshold')
    if (len(X_shape) != 4):
        raise ValueError('Only one channel allowed. Preferably Tmax.')

    return Tmax6
