import numpy as np
from sklearn.metrics import roc_curve
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation
from .Threshold_Model import Threshold_Model
from voxelwise.scoring_utils import cutoff_youdens_j
from voxelwise.channel_normalisation import normalise_channel_by_Tmax4, normalise_channel_by_contralateral

class Campbell_threshold():
    '''
    Model derived from
    Campbell Bruce C.V., Christensen Søren, Levi Christopher R., Desmond Patricia M., Donnan Geoffrey A., Davis Stephen M., et al. Cerebral Blood Flow Is the Optimal CT Perfusion Parameter for Assessing Infarct Core. Stroke. 2011 Dec 1;42(12):3435–40.
    '''
    def __init__(self, rf, threshold = 'train'):
        self.rf = np.max(rf)
        self.train_threshold = np.nan
        self.fixed_threshold = threshold
        print('Using threshold:', self.fixed_threshold)

        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')

    def normalise_channel(self, all_channel_data, data_positions, channel):
        '''
        :param all_channel_data: image input data for all subjects in form of an np array [i, c]
        :param data_positions: boolean map in original space with 1 as placeholder for data points
        :param channel: channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV
        :return: normalised channel by contralateral side
        '''
        normalised_by_contralateral = normalise_channel_by_contralateral(all_channel_data, data_positions, channel)
        return normalised_by_contralateral[1]

    def fit(self, X_train, y_train, train_batch_positions):
        CBF_normalised_byContralateral = self.normalise_channel(X_train, train_batch_positions, 1)

        # As there is an inverse relation between CBF and voxel infarction, inverse CBF before ROC analysis
        inverse_CBF_normalised_byContralateral = -1 * CBF_normalised_byContralateral

        fpr, tpr, thresholds = roc_curve(y_train, inverse_CBF_normalised_byContralateral)
        # get optimal cutOff (inverse again)
        self.train_threshold = -1 * cutoff_youdens_j(fpr, tpr, thresholds)
        print('Training threshold:', self.train_threshold)

        return self

    def predict_proba(self, data, data_position_indices):
        CBF_normalised_byContralateral = self.normalise_channel(data, data_position_indices, 1)

        if self.fixed_threshold == 'train':
            threshold = self.train_threshold
        else: threshold = self.fixed_threshold

        print('Prediction threshold:', threshold)
        thresholded_voxels = np.zeros(data[..., 0].shape)
        # Regions where CBF < 30% of healthy tissue (controlateral)
        thresholded_voxels[CBF_normalised_byContralateral < threshold] = 1

        # Return untresholded data
        # return np.squeeze(-1 * CBF_normalised_byContralateral)
        # Return thresholded prediction
        return np.squeeze(thresholded_voxels)

def Campbell_Model_Generator(X_shape, feature_scaling, pre_smoothing, threshold='train'):
    """
    Model Generator for Campbell threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c)
        feature_scaling: boolean

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the thresholds')
    if not pre_smoothing:
        raise ValueError('Smoothing should be applied beforehand')

    if (len(X_shape) != 5):
        raise ValueError('All channels needed.')

    class custom_Campbell_model(Threshold_Model):
        def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 0):
            super().__init__(fold_dir, fold_name, model = Campbell_threshold(rf, threshold))
            if (n_channels != 4):
                raise Exception('Campbell Threshold model only works with all channels (in the right order)')

        @staticmethod
        def hello_world():
            print('Custom Campbell Model')
            print('Feature scaling is not allowed, as it changes the threshold')
            print('Channels have to respect this order : 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV')

        @staticmethod
        def get_settings():
            return "Custom Campbell threshold model"

    return custom_Campbell_model
