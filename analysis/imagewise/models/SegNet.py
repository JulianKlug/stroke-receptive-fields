import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import UpSampling3D, Conv3D, Dropout, BatchNormalization, MaxPooling3D, ZeroPadding3D, Cropping3D
from tensorflow.keras.callbacks import TensorBoard
from .metrics import weighted_dice_coefficient, dice_coefficient, tversky_coeff


class SegNet:

    def __init__(self, input_shape, model=None):
        self.model_name = 'SegNet'

        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.kernel_size = 3
        self.n_epochs = 1
        self.batch_size = 32
        self.dropout_rate = 0.5

        self.evaluation_threshold = 0.5

        if model is None:
            self.model = segnetwork(input_shape, self.kernel_size, self.dropout_rate)

            self.model.compile(loss=self.loss,
                          optimizer=self.optimizer,
                          metrics=[
                          weighted_dice_coefficient,
                          dice_coefficient,
                          tversky_coeff,
                          'acc',
                          'mse',])
        else:
            self.model = model

        self.model.summary()

        self.settings = {
            'model_name': self.model_name,
            'optimizer': self.optimizer,
            'loss_function': self.loss,
            'kernel_size': self.kernel_size,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
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

    def train(self, x_train, y_train, mask_train, log_dir, epochs=None):
        y_train = np.expand_dims(y_train, axis=-1)
        tensorboard_callback = TensorBoard(log_dir = log_dir)
        if epochs is None: epochs = self.n_epochs
        if epochs is 0:
            return self, {}
        history = self.model.fit(x_train, y_train, validation_split=0.15,
                                 batch_size = self.batch_size, epochs = epochs,
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

    @staticmethod
    def load_model(input_shape, model_path):
        dependencies = {
            'weighted_dice_coefficient': weighted_dice_coefficient,
            'dice_coefficient': dice_coefficient,
            'tversky_coeff': tversky_coeff
        }

        trained_model = load_model(model_path, custom_objects=dependencies)
        model = SegNet(input_shape, model=trained_model)
        return model



def segnetwork(img_shape, kernel_size, Dropout_rate):
    model = Sequential()

    model.add(ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)), input_shape=img_shape))

    # Encoder Layers
    model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))

    # model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv3D(256, kernel_size, activation='relu', padding='same'))
    # model.add(MaxPooling3D((2, 2, 2), padding='same'))
    # model.add(Dropout(Dropout_rate))

    # Decoder Layers
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Dropout(Dropout_rate))

    model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Dropout(Dropout_rate))

    # model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv3D(32, kernel_size, activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(UpSampling3D((2, 2, 2)))
    # model.add(Dropout(Dropout_rate))

    model.add(Cropping3D(cropping=((0, 1), (0, 1), (0, 1))))

    model.add(Conv3D(2,1, activation='relu', padding='same'))  #try this
    model.add(Conv3D(1, 1, activation='sigmoid', padding='same'))



    return model