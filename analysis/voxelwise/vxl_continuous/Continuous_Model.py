import os
import numpy as np

class Continuous_Model():
    """
    This model predicts any voxel by using a threshold
    """

    def __init__(self, fold_dir, fold_name, model = None):
        super(Continuous_Model, self).__init__()
        self.model = model
        self.trained_model = None

        self.X_train = None
        self.y_train = None
        self.train_voxel_index = 0
        self.X_test = None
        self.y_test = None
        self.test_voxel_index = 0

        self.fold_dir = fold_dir
        self.fold_name = fold_name

    @staticmethod
    def hello_world():
        print('Continuous Model')

    @staticmethod
    def get_settings():
        return {}

    def initialise_train_data(self, n_datapoints, data_point_dimensions, n_images, image_spatial_dimensions):
        '''
        :param n_datapoints: number of individual training sample units
        :param data_point_dimensions: number of features of each sample
        :param n_images: number of whole images in train dataset
        :param image_spatial_dimensions: (x, y, z) of a whole image
        '''
        self.X_train = np.empty([np.sum(n_datapoints), data_point_dimensions])
        self.y_train = np.empty(np.sum(n_datapoints))
        # position indices keep track of where the data was stored spatially
        self.position_indices_train = np.zeros(((n_images,) + image_spatial_dimensions))
        self.train_voxel_index = 0; self.train_image_index = 0

    def add_train_data(self, batch_X_train, batch_y_train, batch_positional_indices, batch_n_images):
        """
        Add a batch of training data to the whole training data pool

        Args:
            batch_X_train: batch of training data
            batch_y_train: batch of training labels
            batch_positional_indices: indices that the training data had in the whole entity (ie. image)
            batch_n_images: number of images in this batch
        """
        self.X_train[self.train_voxel_index : self.train_voxel_index + batch_X_train.shape[0], :] = batch_X_train
        self.y_train[self.train_voxel_index : self.train_voxel_index + batch_y_train.shape[0]] = batch_y_train
        self.position_indices_train[self.train_image_index : self.train_image_index + batch_n_images] \
            = batch_positional_indices.reshape((batch_n_images,) + self.position_indices_train[0].shape)
        self.train_voxel_index += batch_X_train.shape[0]; self.train_image_index += batch_n_images

    def initialise_test_data(self, n_datapoints, data_point_dimensions, n_images, image_spatial_dimensions):
        '''
        :param n_datapoints: number of individual testing sample units
        :param data_point_dimensions: features of each sample
        :param n_images: number of whole images in test dataset
        :param image_spatial_dimensions: (x, y, z) of a whole image
        '''
        self.X_test = np.empty([np.sum(n_datapoints), data_point_dimensions])
        self.y_test = np.empty(np.sum(n_datapoints))
        # position indices keep track of where the data was stored spatially
        self.position_indices_test = np.zeros(((n_images,) + image_spatial_dimensions))
        self.test_voxel_index = 0; self.test_image_index = 0

    def add_test_data(self, batch_X_test, batch_y_test, batch_positional_indices, batch_n_images):
        """
        Add a batch of testing data to the whole testing data pool
        """
        self.X_test[self.test_voxel_index : self.test_voxel_index + batch_X_test.shape[0], :] = batch_X_test
        self.y_test[self.test_voxel_index : self.test_voxel_index + batch_y_test.shape[0]] = batch_y_test
        self.position_indices_test[self.test_image_index : self.test_image_index + batch_n_images] \
            = batch_positional_indices.reshape((batch_n_images,) + self.position_indices_test[0].shape)
        self.test_voxel_index += batch_X_test.shape[0]; self.test_image_index += batch_n_images


    def train(self):
        """
        Train the model on the training data that is available
        :return: trained_model
        :return: trained_threshold - threshold to use on predicted probabilities
        :return: evals - array of train evaluation metrics
        """
        self.trained_model = self.model.fit(self.X_train, self.y_train, self.position_indices_train)
        return self.trained_model, np.nan, []

    def predict(self, data, data_position_indices):
        model = self.trained_model
        if not self.trained_model:
            print('Model has not been trained. Prediction may be hazardous')
            model = self.model
        probas_ = model.predict_proba(data, data_position_indices)
        return probas_

    def predict_test_data(self):
        probas_ = self.predict(self.X_test, self.position_indices_test)
        return probas_

    def get_test_labels(self):
        return self.y_test
