import numpy as np
from voxelwise.dimension_utils import reconstruct_image
from .Continuous_Model import Continuous_Model


class individually_normalized_marker():
    def __init__(self, rf, inverse_relation, remove_outliers = False):
        self.rf = np.max(rf)
        self.inverse_relation = inverse_relation
        self.remove_outliers = remove_outliers

    def fit(self, X_train, y_train, train_batch_positions):
        if self.rf != 0:
            raise ValueError('Model only valid for Rf = 0.')
        if self.inverse_relation:
            X_train =  -1 * X_train

        return self

    def predict_proba(self, data, data_position_indices):
        if not self.rf == 0:
            raise ValueError('Model only valid for Rf = 0.')
        if self.inverse_relation:
            data = -1 * data

        reconstructed = reconstruct_image(data, data_position_indices, data.shape[-1])
        prediction = np.empty(data_position_indices.shape)
        for index, subject_data in enumerate(reconstructed):
            if self.remove_outliers:
                clipped_data = np.clip(subject_data, np.percentile(subject_data, 0.1), np.percentile(subject_data, 99.9))
                subject_max = np.max(clipped_data)
                subject_min = np.min(clipped_data)
                norm_data = np.squeeze((subject_data - subject_min) / (subject_max - subject_min))
                clipped_data = np.clip(norm_data, 0, 1)
                prediction[index] = clipped_data
            else:
                subject_max = np.max(subject_data)
                subject_min = np.min(subject_data)
                prediction[index] = np.squeeze((subject_data - subject_min) / (subject_max - subject_min))

        return np.squeeze(prediction[data_position_indices == 1].reshape(-1))

def Normalized_marker_Model_Generator(X_shape, feature_scaling, normalisation_mode = 0, inverse_relation = False):
    """
    Model Generator for custom threshold models.
    Verifies if feature_scaling is off, and only 1 metric is used.
    Args:
        X_shape: expected to be (n, x, y, z, c), where c = 1
        feature_scaling: boolean
        normalisation_mode: individual vs. population trained normalisation
            - if 0: normalise every image individually (nothing is trained)
            - if 1: normalise every image by trained population values
        inverse_relation: boolean, if True find the threshold UNDER wich to predict true

    Returns: result dictionary
    """
    if (feature_scaling):
        raise ValueError('Feature scaling is not allowed, as it changes the threshold')
    if (len(X_shape) != 4):
        raise ValueError('Only one channel allowed.')

    class normalized_marker_model(Continuous_Model):
        def __init__(self, fold_dir, fold_name, n_channels = 1, n_channels_out = 1, rf = 1, model=None):
            if model is None:
                super().__init__(fold_dir, fold_name, model = individually_normalized_marker(rf, inverse_relation))
            else:
                print('Loading pretrained model')
                super().__init__(fold_dir, fold_name, model=model, pretrained=True)

            if (n_channels != 1):
                raise Exception('Normalized marker model only works with one channel.')

        @staticmethod
        def hello_world():
            print('Normalized marker Model')
            print('Feature scaling is not allowed, as it changes the threshold')

        @staticmethod
        def get_settings():
            return "Normalized marker Model"

    return normalized_marker_model
