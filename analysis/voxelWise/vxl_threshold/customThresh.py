import numpy as np
from sklearn.metrics import roc_curve

from vxl_threshold.Threshold_Model import Threshold_Model
from scoring_utils import cutoff_youdens_j

class custom_threshold():
    def __init__(self, rf, fixed_threshold, inverse_relation):
        self.rf = np.max(rf)
        self.train_threshold = np.nan
        self.fixed_threshold = fixed_threshold
        self.inverse_relation = inverse_relation

    def fit(self, X_train, y_train):
        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')
        if self.inverse_relation:
            X_train =  -1 * X_train
        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        # get optimal cutOff
        self.train_threshold = cutoff_youdens_j(fpr, tpr, thresholds)
        print('Training threshold:', self.train_threshold)

        return self

    def predict_proba(self, data):
        if not self.rf == 0:
            raise ValueError('Model only valid for Rf = 0.')
        if self.inverse_relation:
            data =  -1 * data
        # Simple Treshold case
        if self.fixed_threshold: threshold = self.fixed_threshold
        else: threshold = self.train_threshold

        print('Predicting with threshold', threshold)
        tresholded_voxels = np.zeros(data.shape)
        tresholded_voxels[data > threshold] = 1
        return np.squeeze(tresholded_voxels)

def customThreshold_Model_Generator(X_shape, feature_scaling, fixed_threshold = False, inverse_relation = False):
    """
    Model Generator for custom threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c), where c = 1
        feature_scaling: boolean
        fixed_threshold: determine if fixed threshold should be used or if threshold should be determined on training data
            - if False: threshold will be determined from training data
            - if number: given number will be used as threshold (if inverse relation this should be negative)
        inverse_relation: boolean, if True find the threshold UNDER wich to predict true

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the threshold')
    if (len(X_shape) != 4):
        raise ValueError('Only one channel allowed.')
    if inverse_relation and fixed_threshold > 0:
        raise ValueError('Threshold must be negative if relation is inverse')

    class custom_threshold_model(Threshold_Model):
        def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 1):
            super().__init__(fold_dir, fold_name, model = custom_threshold(rf, fixed_threshold, inverse_relation))
            if (n_channels != 1):
                raise Exception('Treshold model only works with one channel.')

        @staticmethod
        def hello_world():
            print('Custom threshold Model')
            print('Feature scaling is not allowed, as it changes the threshold')

        @staticmethod
        def get_settings():
            return "Custom threshold model"

    return custom_threshold_model
