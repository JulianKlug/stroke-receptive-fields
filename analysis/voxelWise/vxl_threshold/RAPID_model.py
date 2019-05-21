import numpy as np
from sklearn.metrics import roc_curve
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation
from vxl_threshold.Threshold_Model import Threshold_Model
from scoring_utils import cutoff_youdens_j

class RAPID_threshold():
    def __init__(self, rf):
        self.rf = np.max(rf)
        self.train_threshold = np.nan
        self.fixed_threshold = 0.40
        print('Using threshold at', self.fixed_threshold)

        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')

    def reconstruct_image(self, data_points, data_positions, n_channels):
        '''
        Reconstruct 2D/3D shape of images with available data filled in (may not be whole image because of sampling)
        :param data_points: data for every available voxel [i, c]
        :param data_positions: (n, x, y, z) boolean array where True marks a given data_point
        :return: reconstructed
        '''
        reconstructed = np.zeros(data_positions.shape + tuple([n_channels]))
        reconstructed[data_positions == 1] = data_points
        return reconstructed

    def normalise_channel_by_Tmax4(self, all_channel_data, data_positions, channel):
        """
        The selected channel is normalised by dividing all its data by the median value
        in healthy tissue. Healthy tissue is defined as the tissue where Tmax < 4s.

        Args:
            all_channel_data : image input data for all subjects in form of an np array [..., c]
                Data should not be scaled beforehand!
            data_positions: boolean map in original space with 1 as placeholder for data points
            mask: boolean array differentiating brain from background [...]
            channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

        Returns: normalised_channel
        """
        # reconstruct to be able to separate individual subjects
        reconstructed = self.reconstruct_image(all_channel_data, data_positions, all_channel_data.shape[-1])
        channel_to_normalise = reconstructed[..., channel]
        Tmax = reconstructed[..., 0]
        normalised_channel = np.zeros(channel_to_normalise.shape)
        for subj in range(data_positions.shape[0]):
            # todo detect cases where Tmax is scaled x10
            masked_healthy_image_voxels = channel_to_normalise[subj][np.all([data_positions[subj] == 1, Tmax[subj] < 4], axis=0)]
            # clip to remove extremes that falsify results
            masked_healthy_image_voxels = np.clip(masked_healthy_image_voxels, np.percentile(masked_healthy_image_voxels, 1), np.percentile(masked_healthy_image_voxels, 99))
            median_channel_healthy_tissue = np.median(masked_healthy_image_voxels)
            normalised_channel[subj] = np.divide(channel_to_normalise[subj], median_channel_healthy_tissue)

        image_normalised_channel = normalised_channel
        flat_normalised_channel = normalised_channel[data_positions == 1].reshape(-1)
        return image_normalised_channel, flat_normalised_channel

    def normalise_channel_by_contralateral(self, all_channel_data, data_positions, channel):
        '''
        Normalise a channel by dividing every voxel by the median value of contralateral side
        :param all_channel_data: image input data for all subjects in form of an np array [i, c]
        :param data_positions: boolean map in original space with 1 as placeholder for data points
        :param channel: channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV
        :return: (image_normalised_channel, flat_normalised_channel) - normalised channel in 3D and flat form
        '''
        # Recover 3D shape of data
        reconstructed_data = self.reconstruct_image(all_channel_data, data_positions, all_channel_data.shape[-1])
        channel_to_normalise_data = reconstructed_data[... ,channel]
        normalised_channel = np.zeros(channel_to_normalise_data.shape, dtype=np.float64)
        x_center = channel_to_normalise_data.shape[1] // 2

        for subj in range(channel_to_normalise_data.shape[0]):
            # normalise left side
            right_side = channel_to_normalise_data[subj][x_center:][data_positions[subj][x_center:] == 1]
            clipped_right_side = np.clip(right_side, np.percentile(right_side, 1), np.percentile(right_side, 99))
            right_side_median = np.nanmedian(clipped_right_side)
            normalised_channel[subj][:x_center] = np.divide(channel_to_normalise_data[subj][:x_center], right_side_median)

            # normalise right side
            left_side = channel_to_normalise_data[subj][:x_center][data_positions[subj][:x_center] == 1]
            clipped_left_side = np.clip(left_side, np.percentile(left_side, 1), np.percentile(left_side, 99))
            left_side_median = np.nanmedian(clipped_left_side)
            normalised_channel[subj][x_center:] = np.divide(channel_to_normalise_data[subj][x_center:], left_side_median)

        image_normalised_channel = normalised_channel
        flat_normalised_channel = normalised_channel[data_positions == 1].reshape(-1)
        return image_normalised_channel, flat_normalised_channel

    def normalise_channel(self, all_channel_data, data_positions, channel):
        '''
        :param all_channel_data: image input data for all subjects in form of an np array [i, c]
        :param data_positions: boolean map in original space with 1 as placeholder for data points
        :param channel: channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV
        :return: normalised channel by Tmax and by contralateral side
        '''
        normalised_by_Tmax4 = self.normalise_channel_by_Tmax4(all_channel_data, data_positions, channel)
        normalised_by_contralateral = self.normalise_channel_by_contralateral(all_channel_data, data_positions, channel)
        return normalised_by_Tmax4[1], normalised_by_contralateral[1]

    def smooth_prediction(self, prediction, data_positions):
        # Recover 3D shape of data
        reconstructed_pred = self.reconstruct_image(prediction.reshape(-1, 1), data_positions, 1)
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
