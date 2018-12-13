import os, sys
sys.path.insert(0, '../')
import data_loader
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from matplotlib import pyplot as plt

batch_size = 32
kernel_size = 3
EPOCHS = 10

main_dir = '/Users/julian/master/data/from_Server'
data_dir = os.path.join(main_dir, '')
main_output_dir = os.path.join(main_dir, 'models')
main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT, MASKS = data_loader.load_saved_data(data_dir)
CLIN = None

def twolayernetwork(img_shape, kernel_size):
    model = Sequential()

    model.add(Conv3D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv3D(1, kernel_size, activation='sigmoid', padding='same'))
    model.add(BatchNormalization())

    generaltheoryofgravity = [model, 'twolayernetwork']

    return generaltheoryofgravity

x_train, y_train = IN, OUT
y_train = np.expand_dims(y_train, axis=5)
n_train_subj, nx, ny, nz, nc = x_train.shape
print('lolilol', x_train.shape, y_train.shape)

model, model_name = twolayernetwork((nx, ny, nz, nc), kernel_size)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.15, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)
print(history.history.keys())
#
# # Plot training & validation accuracy values
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig(os.path.join(main_output_dir, model_name + str(kernel_size) + 'accuracy_values.png'))
#
# plt.figure()
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig(os.path.join(main_output_dir, model_name + str(kernel_size) + 'loss_values.png'))
#

y_pred = model.predict(x_train)

# plt.figure(1)
# plt.imshow(x_train[0])
# plt.show()


num_images = 3

y = np.zeros((num_images, nx, ny, nz, 1))
y[:,:,:,:] = y_pred
plt.imshow(y[0,:,:,nz//2,0])
plt.show()
threshold, upper, lower = 0.5, 1, 0
y = np.where(y > threshold, upper, lower)
y= 255*y

# for i in range(0,num_images):
#    plt.figure()
#    plt.imshow(y[i])
#    plt.show()


plt.figure(figsize=(ny, nx))

for i in range(0,num_images):
   # plot original image
   ax = plt.subplot(2, num_images, i + 1)
   plt.imshow(x_train[i,:,:,nz//2,0])

   # plot reconstructed image
   ax = plt.subplot(2, num_images, num_images + i + 1)
   plt.imshow(y[i,:,:,nz//2,0])

plt.savefig(os.path.join(main_output_dir, model_name + str(kernel_size) + 'figure.png'), bbox_inches='tight')
