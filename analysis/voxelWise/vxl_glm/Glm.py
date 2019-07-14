import sys
sys.path.insert(0, '../')
import numpy as np


class Glm():
    """
    """

    def __init__(self, fold_dir, fold_name, model = None):
        super(Glm, self).__init__()
        self.model = model
        self.trained_model = None

        self.X_train = None
        self.y_train = None
        self.train_index = 0
        self.X_test = None
        self.y_test = None
        self.test_index = 0

        self.fold_dir = fold_dir
        self.fold_name = fold_name

    @staticmethod
    def hello_world():
        print('GLM Model')

    @staticmethod
    def get_settings():
        return {}

    def initialise_train_data(self, n_datapoints, data_dimensions, n_images = None, image_spatial_dimensions = None):
        self.X_train = np.empty([np.sum(n_datapoints), data_dimensions])
        self.y_train = np.empty(np.sum(n_datapoints))
        self.train_index = 0

    def add_train_data(self, batch_X_train, batch_y_train, batch_positional_indices = None, batch_n_images = None):
        """
        Add a batch of training data to the whole training data pool

        Args:
            batch_X_train: batch of training data
            batch_y_train: batch of training labels
        """
        self.X_train[self.train_index : self.train_index + batch_X_train.shape[0], :] = batch_X_train
        self.y_train[self.train_index : self.train_index + batch_y_train.shape[0]] = batch_y_train
        self.train_index += batch_X_train.shape[0]

    def initialise_test_data(self, n_datapoints, data_dimensions,  n_images = None, image_spatial_dimensions = None):
        self.X_test = np.empty([np.sum(n_datapoints), data_dimensions])
        self.y_test = np.empty(np.sum(n_datapoints))
        self.test_index = 0

    def add_test_data(self, batch_X_test, batch_y_test, batch_positional_indices = None, batch_n_images = None):
        """
        Add a batch of testing data to the whole testing data pool
        All testing data is saved in a svmlight file
        """
        self.X_test[self.test_index : self.test_index + batch_X_test.shape[0], :] = batch_X_test
        self.y_test[self.test_index : self.test_index + batch_y_test.shape[0]] = batch_y_test
        self.test_index += batch_X_test.shape[0]

    def train(self):
        """
        Train the model on the training data that is available
        :return: trained_model
        :return: trained_threshold - threshold to use on predicted probabilities
        :return: evals - array of train evaluation metrics
        """
        self.trained_model = self.model.fit(self.X_train, self.y_train)

        # default threshold for logistic regression is 0.5, determining it through analysis of test data exacerbates overfitting
        # https://stackoverflow.com/questions/31417487/sklearn-logisticregression-and-changing-the-default-threshold-for-classification?rq=1
        self.trained_threshold = 0.5

        return self.trained_model, self.trained_threshold, []

    def predict(self, data, data_position_indices = None):
        probas_ = self.trained_model.predict_proba(data)
        return probas_[:, 1]

    def predict_test_data(self):
        probas_ = self.predict(self.X_test)
        return probas_

    def get_test_labels(self):
        return self.y_test
