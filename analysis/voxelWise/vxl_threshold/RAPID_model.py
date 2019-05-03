import numpy as np
from sklearn.metrics import roc_curve

from vxl_threshold.Threshold_Model import Threshold_Model
from scoring_utils import cutoff_youdens_j

class RAPID_threshold():
    def __init__(self, rf):
        self.rf = np.max(rf)
        self.train_threshold = np.nan
        self.fixed_threshold = 0.3

        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')

    def normalise_channel(self, all_channel_data, mask, channel):
        """
        The selected channel is normalised by dividing all its data by the median value
        in healthy tissue. Healthy tissue is defined as the tissue where Tmax < 4s.

        Args:
            all_channel_data : image input data for all subjects in form of an np array [..., c]
                Data should not be scaled beforehand!
            mask: boolean array differentiating brain from brackground [...]
            channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

        Returns: normalised_channel
        """
        channel_to_normalise = all_channel_data[... ,channel]
        Tmax = all_channel_data[..., 0]
        normalised_channel = np.empty(channel_to_normalise.shape)

        masked_healthy_image_voxels = channel_to_normalise[np.all([mask, Tmax < 4], axis = 0)]
        median_channel_healthy_tissue = np.median(masked_healthy_image_voxels)
        normalised_channel = np.divide(channel_to_normalise, median_channel_healthy_tissue)

        return normalised_channel

    def threshold_Tmax6(self, data):
        # Get penumbra mask
        Tmax = data[..., 0]
        tresholded_voxels = np.zeros(Tmax.shape)
        # penumbra (Tmax > 6)
        tresholded_voxels[Tmax > 6] = 1
        return np.squeeze(tresholded_voxels)

    def fit(self, X_train, y_train):
        Tmax = X_train[..., 0]
        normalised_CBF = self.normalise_channel(X_train, np.ones(Tmax.shape), 1)
        penumbra_indeces = np.where(Tmax > 6) # define penumbra
        penumbra_normalised_CBF = normalised_CBF[penumbra_indeces]
        fpr, tpr, thresholds = roc_curve(y_train[penumbra_indeces], penumbra_normalised_CBF)
        # get optimal cutOff
        self.train_threshold = cutoff_youdens_j(fpr, tpr, thresholds)
        print('Training threshold:', self.train_threshold)

        return self

    def predict_proba(self, data):
        Tmax = data[..., 0]
        normalised_CBF = self.normalise_channel(data, np.ones(Tmax.shape), 1)

        threshold = self.fixed_threshold
        tresholded_voxels = np.zeros(Tmax.shape)
        # Parts of penumbra (Tmax > 6) where CBF < 30% of healthy tissue)
        tresholded_voxels[(normalised_CBF < threshold) & (Tmax > 6)] = 1
        return np.squeeze(tresholded_voxels)

def RAPID_Model_Generator(X_shape, feature_scaling):
    """
    Model Generator for custom threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c)
        feature_scaling: boolean
        fixed_threshold: determine if fixed threshold should be used or if threshold should be determined on training data
            - if False: threshold will be determined from training data
            - if number: given number will be used as threshold (if inverse relation this should be negative)
        inverse_relation: boolean, if True find the threshold UNDER wich to predict true

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the thresholds')
    if (len(X_shape) != 5):
        raise ValueError('All channels needed.')

    class custom_RAPID_model(Threshold_Model):
        def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 0):
            super().__init__(fold_dir, fold_name, model = RAPID_threshold(rf))
            if (n_channels != 4):
                raise Exception('RAPID Treshold model only works with all channels.')

        @staticmethod
        def hello_world():
            print('Custom RAPID Model')
            print('Feature scaling is not allowed, as it changes the threshold')
            print('Channels have to respect this order : 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV')

        @staticmethod
        def get_settings():
            return "Custom threshold model"

    return custom_RAPID_model
