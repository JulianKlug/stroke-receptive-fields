import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model

def receptive_field_log_model(input_data_array, output_data_array, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions

    # Declare final arrays going into model
    input = []
    output = []

    # Iterate through all images
    for i in range(0, len(input_data_array)):

        input_data = input_data_array[i]
        output_data = output_data_array[i]

        if (input_data.shape != output_data.shape):
            raise ValueError('Input and output do not have the same shape.', input_data.shape, output_data.shape)

        n_x, n_y, n_z = input_data.shape
        print(input_data.shape)
        print('output', output_data.shape)

        # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
        padding = max([rf_x, rf_y, rf_z])
        padded_input_data = np.pad(input_data[:,:,:], pad_width=padding, mode='constant', constant_values=0);

        # iterate through all pixels in image and put receptive field as input for this output pixel
        for x, y, z in itertools.product(range(n_x),
                                         range(n_y),
                                         range(n_z)):
        # for x, y, z in itertools.product(range(2),
        #                                  range(2),
        #                                  range(2)):
            px = x + padding; py = y + padding; pz = z + padding;

            output_voxel = np.array([output_data[x,y,z]])
            output.append(output_voxel)

            input_field = padded_input_data[
            px - rf_x : px + rf_x + 1,
            py - rf_y : py + rf_y + 1,
            pz - rf_z : pz + rf_z + 1
            ]
            linear_input = np.reshape(input_field, input_field.size)
            input.append(linear_input)


    input = np.array(input)
    output = np.array(output)

    print(input.shape)
    print(output.shape)

    # Create linear regression object
    log_reg = linear_model.LogisticRegression()

    # Train the model using the training sets
    log_reg.fit(input, output)

    # The coefficients
    print('Coefficients: \n', log_reg.coef_)
    print('Intercept: \n', log_reg.intercept_)

    return log_reg

def reconstruct(input, model, receptive_field_dimensions):
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions

    n_x, n_y, n_z = input.shape
    print(input.shape)

    output = np.zeros(input.shape)
    print(output.shape)

    # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
    padding = max([rf_x, rf_y, rf_z])
    padded_input_data = np.pad(input[:,:,:], pad_width=padding, mode='constant', constant_values=0);

    # iterate through all pixels in image and put receptive field as input for this output pixel
    for x, y, z in itertools.product(range(n_x),
                                     range(n_y),
                                     range(n_z)):

        px = x + padding; py = y + padding; pz = z + padding;

        input_field = padded_input_data[
        px - rf_x : px + rf_x + 1,
        py - rf_y : py + rf_y + 1,
        pz - rf_z : pz + rf_z + 1
        ]

        linear_input = np.reshape(input_field, input_field.size)
        linear_input = np.reshape(linear_input, (1, -1)) # as this is only one sample

        output[x, y, z] = model.predict(linear_input)

    return output
