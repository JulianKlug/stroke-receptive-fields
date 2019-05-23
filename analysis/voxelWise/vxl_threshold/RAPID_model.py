import numpy as np
from sklearn.metrics import roc_curve
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation
from vxl_threshold.Threshold_Model import Threshold_Model
from scoring_utils import cutoff_youdens_j
from dimension_utils import reconstruct_image
from channel_normalisation import normalise_channel_by_Tmax4, normalise_channel_by_contralateral

class RAPID_threshold():
    def __init__(self, rf):
        self.rf = np.max(rf)
        self.train_threshold = np.nan
        self.fixed_threshold = 0.40
        print('Using threshold at', self.fixed_threshold)

        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')

    def normalise_channel(self, all_channel_data, data_positions, channel):
        '''
        :param all_channel_data: image input data for all subjects in form of an np array [i, c]
        :param data_positions: boolean map in original space with 1 as placeholder for data points
        :param channel: channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV
        :return: normalised channel by Tmax and by contralateral side
        '''
        normalised_by_Tmax4 = normalise_channel_by_Tmax4(all_channel_data, data_positions, channel)
        normalised_by_contralateral = normalise_channel_by_contralateral(all_channel_data, data_positions, channel)
        return normalised_by_Tmax4[1], normalised_by_contralateral[1]

    def smooth_prediction(self, prediction, data_positions):
        # Recover 3D shape of data
        reconstructed_pred = reconstruct_image(prediction.reshape(-1, 1), data_positions, 1)
        smooth_pred = np.zeros(reconstructed_pred.shape[:4])
        structure = np.ones((1, 1, 1), dtype=np.int)
        for subj in range(reconstructed_pred.shape[0]):
            smooth_pred[subj] = binary_erosion(np.squeeze(reconstructed_pred[subj]), structure)
            smooth_pred[subj] = binary_dilation(smooth_pred[subj], structure)
            smooth_pred[subj] = binary_closing(smooth_pred[subj], np.ones((4, 4, 4), dtype=np.int))
        flat_smooth_pred = smooth_pred[data_positions == 1].reshape(-1)
        return flat_smooth_pred


    def threshold_Tmax6(self, data):
        # Get penumbra mask
        Tmax = data[..., 0]
        tresholded_voxels = np.zeros(Tmax.shape)
        # penumbra (Tmax > 6) without extremes
        tresholded_voxels[(Tmax > 6) & (Tmax < np.percentile(Tmax, 99))] = 1 # define penumbra
        return np.squeeze(tresholded_voxels)

    def fit(self, X_train, y_train, train_batch_positions):
        penumbra_indices = np.where(self.threshold_Tmax6(X_train) == 1)
        CBF_normalised_byTmax4, CBF_normalised_byContralateral = self.normalise_channel(X_train, train_batch_positions, 1)

        # todo This training is just for show
        penumbra_normalised_CBF = CBF_normalised_byTmax4[penumbra_indices]
        fpr, tpr, thresholds = roc_curve(y_train[penumbra_indices], penumbra_normalised_CBF)
        # get optimal cutOff
        self.train_threshold = cutoff_youdens_j(fpr, tpr, thresholds)
        print('Training threshold:', self.train_threshold)

        return self

    def predict_proba(self, data, data_position_indices):
        penumbra = self.threshold_Tmax6(data) == 1

        CBF_normalised_byTmax4, CBF_normalised_byContralateral = self.normalise_channel(data, data_position_indices, 1)

        threshold: float = self.fixed_threshold
        tresholded_voxels = np.zeros(data[..., 0].shape)
        # Parts of penumbra (Tmax > 6) where CBF < 30% of healthy tissue (controlateral or region where Tmax < 4s)
        tresholded_voxels[(CBF_normalised_byContralateral < threshold) & (penumbra)] = 1
        tresholded_voxels[(CBF_normalised_byTmax4 < threshold) & (penumbra)] = 1
        smoothed_tresholded_voxels = self.smooth_prediction(tresholded_voxels, data_position_indices)

        return np.squeeze(smoothed_tresholded_voxels)

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
