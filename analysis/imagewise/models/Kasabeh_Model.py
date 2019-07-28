import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from .metrics import weighted_dice_coefficient, dice_coefficient, tversky_coeff


class Kasabeh_Model:

    def __init__(self, input_shape):
        self.model_name = 'Kasabeh_Model'

        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.n_epochs = 10
        self.batch_size = 32

        self.evaluation_threshold = 0.5

        self.model = Sequential()
        self.model.add(Dense(32, activation='tanh', input_shape=input_shape))
        self.model.add(Dense(1, activation='tanh'))
        self.model.add(Dense(1, activation='tanh'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.summary()
        self.model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=[
                      weighted_dice_coefficient,
                      dice_coefficient,
                      tversky_coeff,
                      'acc',
                      'mse',])

        self.settings = {
            'model_name': self.model_name,
            'optimizer': self.optimizer,
            'loss_function': self.loss,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'architecture': self.model.to_json(),
            # for now brain masking is not used
            'used_brain_masking': False,
            # todo undersampling has to be done at model level
            'used_undersampling': False,
            'input_pre_normalisation': True
        }

    def hello_world(self):
        print('This is', self.model_name)

    def get_settings(self):
        return self.settings

    def get_threshold(self):
        return self.evaluation_threshold

    def train(self, x_train, y_train, mask_train, log_dir):
        y_train = np.expand_dims(y_train, axis=-1)
        tensorboard_callback = TensorBoard(log_dir = log_dir)
        history = self.model.fit(x_train, y_train, validation_split=0.15,
                                 batch_size = self.batch_size, epochs = self.n_epochs,
                                 verbose=1, callbacks = [tensorboard_callback])
        train_eval = {
            'train': { 'loss': history.history['loss'], 'acc': history.history['acc'] },
            'eval': { 'loss': history.history['val_loss'], 'acc': history.history['val_acc'] }
            }
        return self, train_eval

    def predict(self, data, mask_data):
        probas_ = self.model.predict(data)
        probas_ = np.squeeze(probas_) # reduce empty dimensions
        return probas_

    def save(self, save_path):
        self.model.save(save_path)
