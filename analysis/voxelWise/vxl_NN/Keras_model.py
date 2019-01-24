import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, Dense, Flatten
from keras.layers.normalization import BatchNormalization

EPOCHS = 10

class Keras_model():
    """
    """

    def __init__(self, fold_dir, fold_name, model = None, n_channels = 4, rf_dim = 1, n_epochs = 100):
        super(Keras_model, self).__init__()
        self.model = model
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.n_channels = n_channels
        self.rf_width = 2 * np.max(rf_dim) + 1
        self.n_epochs = n_epochs
        self.batch_size = 32

        self.X_train = None
        self.y_train = None
        self.train_index = 0
        self.X_test = None
        self.y_test = None
        self.test_index = 0

        self.fold_dir = fold_dir
        self.fold_name = fold_name

        model.summary()
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

    @staticmethod
    def hello_world():
        print('Keras NN Model')

    @staticmethod
    def get_settings():
        return {}

    def initialise_train_data(self, n_datapoints, data_dimensions):
        self.X_train = np.empty([np.sum(n_datapoints), self.rf_width, self.rf_width, self.rf_width, self.n_channels])
        self.y_train = np.empty(np.sum(n_datapoints))
        self.train_index = 0

    def add_train_data(self, batch_X_train, batch_y_train):
        """
        Add a batch of training data to the whole training data pool

        Args:
            batch_X_train: batch of training data
            batch_y_train: batch of training labels
        """
        batch_X_train = batch_X_train.reshape(-1, self.rf_width, self.rf_width, self.rf_width, self.n_channels)
        self.X_train[self.train_index : self.train_index + batch_X_train.shape[0], :] = batch_X_train
        self.y_train[self.train_index : self.train_index + batch_y_train.shape[0]] = batch_y_train
        self.train_index += batch_X_train.shape[0]

    def initialise_test_data(self, n_datapoints, data_dimensions):
        self.X_test = np.empty([np.sum(n_datapoints), self.rf_width, self.rf_width, self.rf_width, self.n_channels])
        self.y_test = np.empty(np.sum(n_datapoints))
        self.test_index = 0

    def add_test_data(self, batch_X_test, batch_y_test):
        """
        Add a batch of testing data to the whole testing data pool
        All testing data is saved in a svmlight file
        """
        batch_X_test = batch_X_test.reshape(-1, self.rf_width, self.rf_width, self.rf_width, self.n_channels)
        self.X_test[self.test_index : self.test_index + batch_X_test.shape[0], :] = batch_X_test
        self.y_test[self.test_index : self.test_index + batch_y_test.shape[0]] = batch_y_test
        self.test_index += batch_X_test.shape[0]

    def train(self):
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.15, batch_size = self.batch_size, epochs = self.n_epochs, verbose=1)
        train_eval = {
            'train': { 'loss': history.history['loss'], 'acc': history.history['acc'] },
            'eval': { 'loss': history.history['val_loss'], 'acc': history.history['val_acc'] }
            }
        return self.model, train_eval

    def predict(self, data):
        probas_ =  self.model.predict(data)
        probas_ = np.squeeze(probas_) # reduce single dimension to flat array of predicted voxels
        return probas_

    def predict_test_data(self):
        probas_ = self.predict(self.X_test)
        return probas_

    def get_test_labels(self):
        return self.y_test

class TwoLayerNetwork(Keras_model):

    def __init__(self, fold_dir, fold_name, n_channels = 4, n_channels_out = 1, rf = 1):
        self.model_name = 'TwoLayerNetwork'

        # NN should go over the input only once, taken the whole image patch as input
        image_width = 2 * np.max(rf) + 1
        img_shape = (image_width, image_width, image_width, n_channels)
        kernel_size = (image_width, image_width, image_width)

        self.model = Sequential()
        self.model.add(Conv3D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv3D(1, kernel_size, activation='sigmoid', padding='same'))
        self.model.add(BatchNormalization())
        # As we only have one voxel as output, Flatten is needed to reduce dimensionality
        self.model.add(Flatten())

        super().__init__(fold_dir, fold_name, self.model, n_channels = n_channels, rf_dim = rf, n_epochs = EPOCHS)

    @staticmethod
    def hello_world():
        print('TwoLayerNetwork')

    @staticmethod
    def get_settings():
        return {}
