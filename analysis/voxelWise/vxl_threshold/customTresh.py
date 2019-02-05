import numpy as np
from sklearn.metrics import roc_curve

from vxl_threshold.Treshold_Model import Treshold_Model
from scoring_utils import cutoff_youdens_j

class custom_treshold():
    def __init__(self, rf):
        self.rf = np.max(rf)
        self.train_threshold = np.nan

    def fit(self, X_train, y_train):
        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        # get optimal cutOff
        self.train_threshold = cutoff_youdens_j(fpr, tpr, thresholds)
        print('Training threshold:', self.train_threshold)

        return self


    def predict_proba(self, data):
        if self.rf == 0:
            # Simple Treshold case
            return np.squeeze(data)
        else:
            raise ValueError('Model only valid for Rf = 0.')

class custom_treshold_model(Treshold_Model):
    def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 1):
        super().__init__(fold_dir, fold_name, model = custom_treshold(rf))
        if (n_channels != 1):
            raise Exception('Treshold model only works with one channel.')

    @staticmethod
    def hello_world():
        print('Custom threshold Model')
        print('Feature scaling is not allowed, as it changes the threshold')

    @staticmethod
    def get_settings():
        return "Custom threshold model"

def customTreshold_Model_Generator(X_shape, feature_scaling):
    """
    Model Generator for custom threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c), where c = 1
        feature_scaling: boolean

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the threshold')
    if (len(X_shape) != 4):
        raise ValueError('Only one channel allowed.')

    return custom_treshold_model
