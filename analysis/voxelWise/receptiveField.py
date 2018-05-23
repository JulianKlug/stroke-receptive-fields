import sys
sys.path.insert(0, '../')

from rolling_window import rolling_window

import numpy as np
import itertools

def reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions

    # Declare final lists going into model
    # inputs = []
    # output = []
    #
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )
    # Initialize the inputs array as long as there are voxels in an image, and on the second dimensions as many voxels as there are in a receptive field
    inputs = np.empty((input_data_list[0][:,:,:,0].size, window_d_x * window_d_y * window_d_z * input_data_list[0][0,0,0,:].size))
    output = np.empty(input_data_list[0][:,:,:,0].size)

    # Iterate through all images
    for i in range(0, len(input_data_list)):

        input_data = input_data_list[i]
        output_data = output_data_list[i]
        index = 0

        if (input_data[:,:,:,0].shape != output_data.shape):
            raise ValueError('Input and output do not have the same shape.', input_data[:,:,:,0].shape, output_data.shape)

        n_x, n_y, n_z, n_c = input_data.shape

        # pad the image to allow for an receptive field even at the borders
        padding = max([rf_x, rf_y, rf_z])
        padded_input_data = pad(input_data, padding)

        input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))

        for x, y, z in itertools.product(range(n_x),
                                         range(n_y),
                                         range(n_z)):

            # Reshape to linear input
            linear_input_field = np.reshape(input_fields[x, y, z], input_fields[x, y, z].size)
            inputs[index, :] = linear_input_field
            # inputs.append(linear_input_field)

            output_voxel = np.array([output_data[x, y, z]])
            output[index] = output_voxel
            index += 1
            # output.append(output_voxel)


        # # iterate through all pixels in image and put receptive field as input for this output pixel
        # for x, y, z in itertools.product(range(n_x),
        #                                  range(n_y),
        #                                  range(n_z)):
        # # for x, y, z in itertools.product(range(2),
        # #                                  range(2),
        # #                                  range(2)):
        #     px = x + padding; py = y + padding; pz = z + padding
        #
        #     output_voxel = np.array([output_data[x,y,z]])
        #     output.append(output_voxel)
        #
        #     input_field = padded_input_data[
        #         px - rf_x : px + rf_x + 1,
        #         py - rf_y : py + rf_y + 1,
        #         pz - rf_z : pz + rf_z + 1,
        #         :
        #     ]
        #     linear_input = np.reshape(input_field, input_field.size)
        #     inputs.append(linear_input)


    # inputs = np.array(inputs)
    # output = np.array(output)

    inputs = np.squeeze(inputs)
    output = np.squeeze(output)

    print('Entire dataset. Input shape: ', inputs.shape,
          ' and output shape: ', output.shape)

    return inputs, output


def predict(input_data, model, receptive_field_dimensions):
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions

    n_x, n_y, n_z, n_c = input_data.shape
    print('Predicting from input: ', input_data.shape)

    output = np.zeros([n_x, n_y, n_z])

    # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
    padding = max([rf_x, rf_y, rf_z])
    padded_input_data = pad(input_data, padding)

    # iterate through all pixels in image and put receptive field as input for this output pixel
    for x, y, z in itertools.product(range(n_x),
                                     range(n_y),
                                     range(n_z)):

        px = x + padding; py = y + padding; pz = z + padding

        input_field = padded_input_data[
            px - rf_x : px + rf_x + 1,
            py - rf_y : py + rf_y + 1,
            pz - rf_z : pz + rf_z + 1,
            :
        ]

        linear_input = np.reshape(input_field, input_field.size)
        linear_input = np.reshape(linear_input, (1, -1)) # as this is only one sample

        output[x, y, z] = model.predict_proba(linear_input)[0][1]

    return output

def pad(image_with_channels, padding):
    """
    Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels

    Args:
        image_with_channels: image with n_c channels that is to be padded in its 3 dimensions
        padding: thickness of padding

    Returns:
        padded_data
    """

    n_x, n_y, n_z, n_c = image_with_channels.shape
    padded_image = np.zeros([n_x + 2 * padding, n_y + 2 * padding, n_z + 2 * padding, n_c])
    for c in range(n_c):
        channel = image_with_channels[:, :, :, c]
        padded_channel = np.pad(channel, pad_width=padding, mode='constant', constant_values=0)
        padded_image[:, :, :, c] = padded_channel

    return padded_image
