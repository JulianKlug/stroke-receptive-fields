import sys
sys.path.insert(0, '../')

from rolling_window import rolling_window

import numpy as np
import itertools

def reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    temp = shorter_new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    # temp = short_new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)
    # a_temp, b_temp = short_new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)
    # a_temp2, b_temp2 = new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)
    #
    # print('comparing inpout, output', np.array_equal(a_temp, a_temp2), np.array_equal(b_temp, b_temp2))

    # temp = old_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    return temp

def shorter_new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    n_subjects = len(input_data_list)
    n_receptive_fields = input_data_list[0][:, :, :, 0].size * n_subjects  # ie voxels per image times number of images
    receptive_field_size = window_d_x * window_d_y * window_d_z * input_data_list[0][0,0,0,:].size
    n_x, n_y, n_z, n_c = input_data_list[0].shape
    n_voxels_per_subject = n_x * n_y * n_z


    # pad all images to allow for an receptive field even at the borders
    padding = max([rf_x, rf_y, rf_z])
    padded_data = [pad(x, padding) for x in input_data_list]

    # TODO stack subjects first and then use rolling_window with dimension 1 as 0, (0, window_d_x, window_d_y, window_d_z, 0)
    input_fields = np.stack([rolling_window(x, (window_d_x, window_d_y, window_d_z, 0)) for x in padded_data])

    inputs = input_fields.reshape((n_subjects * n_voxels_per_subject, receptive_field_size))

    outputs = np.stack(output_data_list).reshape(n_subjects * n_voxels_per_subject)

    print('Entire dataset. Input shape: ', inputs.shape,
          ' and output shape: ', outputs.shape)

    return inputs, outputs

def short_new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    # Declare final array going into model
    # Initialize the inputs array as long as there will be receptive fields, and on the second dimensions as many voxels as there are in a receptive field
    n_receptive_fields = input_data_list[0][:, :, :, 0].size * len(input_data_list) # ie voxels per image times number of images
    receptive_field_size = window_d_x * window_d_y * window_d_z * input_data_list[0][0,0,0,:].size
    inputs = np.empty((n_receptive_fields, receptive_field_size))
    outputs = np.empty(n_receptive_fields)

    index = 0


    # Iterate through all images
    for i in range(0, len(input_data_list)):

        input_data = input_data_list[i]
        output_data = output_data_list[i]

        if (input_data[:,:,:,0].shape != output_data.shape):
            raise ValueError('Input and output do not have the same shape.', input_data[:,:,:,0].shape, output_data.shape)

        n_x, n_y, n_z, n_c = input_data.shape
        n_voxels_per_subject = n_x * n_y * n_z

        # pad the image to allow for an receptive field even at the borders
        padding = max([rf_x, rf_y, rf_z])
        padded_input_data = pad(input_data, padding)

        # Create patches centered on all the voxels in the image
        # (do not use the 4th dimension for patches, as this is the perfusion parameter channel)
        input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))
        # Reshape to linear input
        linear_input_fields = input_fields.reshape((n_voxels_per_subject, receptive_field_size))
        inputs[index : index + n_voxels_per_subject] = linear_input_fields

        # Reshape to linear output
        linear_output = output_data.reshape(n_voxels_per_subject)
        outputs[index : index + n_voxels_per_subject] = linear_output

        index += n_voxels_per_subject
    # End for (loop through subjects)

    print('Entire dataset. Input shape: ', inputs.shape,
          ' and output shape: ', outputs.shape)

    return inputs, outputs


def new_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    # Declare final array going into model
    # Initialize the inputs array as long as there will be receptive fields, and on the second dimensions as many voxels as there are in a receptive field
    n_receptive_fields = input_data_list[0][:, :, :, 0].size * len(input_data_list) # ie voxels per image times number of images
    receptive_field_size = window_d_x * window_d_y * window_d_z * input_data_list[0][0,0,0,:].size
    inputs = np.empty((n_receptive_fields, receptive_field_size))
    output = np.empty(n_receptive_fields)

    index = 0


    # Iterate through all images
    for i in range(0, len(input_data_list)):

        input_data = input_data_list[i]
        output_data = output_data_list[i]

        if (input_data[:,:,:,0].shape != output_data.shape):
            raise ValueError('Input and output do not have the same shape.', input_data[:,:,:,0].shape, output_data.shape)

        n_x, n_y, n_z, n_c = input_data.shape

        # pad the image to allow for an receptive field even at the borders
        padding = max([rf_x, rf_y, rf_z])
        padded_input_data = pad(input_data, padding)

        input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))

        # TODO: try returning the whole flattened array?

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


    # inputs = np.squeeze(inputs)
    # output = np.squeeze(output)

    print('Entire dataset. Input shape: ', inputs.shape,
          ' and output shape: ', output.shape)

    # print('IN', inputs[5000:5050])
    # print('OUT', output[5000:5050])

    return inputs, output

def old_reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions

    # Declare final lists going into model
    inputs = []
    output = []

    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    # Iterate through all images
    for i in range(0, len(input_data_list)):

        input_data = input_data_list[i]
        output_data = output_data_list[i]

        if (input_data[:,:,:,0].shape != output_data.shape):
            raise ValueError('Input and output do not have the same shape.', input_data[:,:,:,0].shape, output_data.shape)

        n_x, n_y, n_z, n_c = input_data.shape

        # pad the image to allow for an receptive field even at the borders
        padding = max([rf_x, rf_y, rf_z])
        padded_input_data = pad(input_data, padding)

        # iterate through all pixels in image and put receptive field as input for this output pixel
        for x, y, z in itertools.product(range(n_x),
                                         range(n_y),
                                         range(n_z)):
        # for x, y, z in itertools.product(range(2),
        #                                  range(2),
        #                                  range(2)):
            px = x + padding; py = y + padding; pz = z + padding

            output_voxel = np.array([output_data[x,y,z]])
            output.append(output_voxel)

            input_field = padded_input_data[
                px - rf_x : px + rf_x + 1,
                py - rf_y : py + rf_y + 1,
                pz - rf_z : pz + rf_z + 1,
                :
            ]
            linear_input = np.reshape(input_field, input_field.size)
            inputs.append(linear_input)


    inputs = np.array(inputs)
    output = np.array(output)

    print('Entire dataset. Input shape: ', inputs.shape,
          ' and output shape: ', output.shape)

    # print('IN', inputs[5000:5050])
    # print('OUT', output[5000:5050])


    return inputs, output


def predict(input_data, model, receptive_field_dimensions):
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    n_x, n_y, n_z, n_c = input_data.shape
    print('Predicting from input: ', input_data.shape)

    output = np.zeros([n_x, n_y, n_z])

    # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
    padding = max([rf_x, rf_y, rf_z])
    padded_input_data = pad(input_data, padding)

    # input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))

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


        # Reshape to linear input
        # linear_input = np.reshape(input_fields[x, y, z], input_fields[x, y, z].size)
        linear_input = np.reshape(input_field, input_field.size)

        linear_input = np.reshape(linear_input, (1, -1)) # as this is only one sample

        output[x, y, z] = model.predict_proba(linear_input)[0][1]
        # output[x, y, z] = (1 - model.predict_proba(linear_input)[0][1])

    return output
# TODO use np funciton
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
