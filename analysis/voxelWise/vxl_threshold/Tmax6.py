import numpy as np
from sklearn import linear_model

from vxl_threshold.Treshold_Model import Treshold_Model

class Tmax6_treshold():
    def __init__(self, rf):
        self.rf = np.max(rf)
        self.threshold = 6
        self.combinator = linear_model.LogisticRegression(verbose = 0, max_iter = 1000000000, n_jobs = -1)

    def fit(self, X_train, y_train):
        if self.rf != 0:
            tresholded_voxels = np.zeros(X_train.shape)
            tresholded_voxels[X_train > self.threshold] = 1
            self.combinator.fit(tresholded_voxels, y_train)
        return self


    def predict_proba(self, data):
        tresholded_voxels = np.zeros(data.shape)
        tresholded_voxels[data > self.threshold] = 1
        if self.rf == 0:
            # Simple Treshold case
            return np.squeeze(tresholded_voxels)
        else:
            # log-linear combination of thresholded voxels in receptiveField
            probas_ = self.combinator.predict_proba(tresholded_voxels)
            return probas_[:, 1]

class Tmax6(Treshold_Model):
    def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, model = Tmax6_treshold(rf))
        if (n_channels != 1):
            raise Exception('Tmax6 Treshold model only works with one channel (Preferably Tmax).')

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
