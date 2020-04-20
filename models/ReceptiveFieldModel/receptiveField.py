import numpy as np
from .rolling_window import rolling_window


def reshape_to_receptive_field(input_data_array, output_data_array, receptive_field_dimensions, mask_array = None, include_only = np.NaN, verbose = False) :
    '''
    Transform a given input image into a set of receptive fields
    :param input_data_array: input data
    :param output_data_array:
    :param receptive_field_dimensions: dimensions of the receptive fields in rf (steps from center voxel)
    :param mask_array: boolean array defining brain areas vs. non brain areas
    :param include_only: percentage of data to use
    :param verbose:
    :return:
    '''

    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z = 2 * np.array(receptive_field_dimensions) + 1
    n_x, n_y, n_z, n_c = input_data_array[0].shape

    if verbose:
        print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    n_subjects = input_data_array.shape[0]
    n_receptive_fields = input_data_array[0, :, :, :, 0].size * n_subjects  # ie voxels per image times number of images
    # a receptive field is a 3D patch with n_c channels (then flattened)
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c
    n_voxels_per_subject = n_x * n_y * n_z

    # pass mask array as last channel to apply rf on it as well
    if mask_array is not None:
        # add the subject and channel dimension
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=-1)
        # add mask as last channel
        input_data_array = np.concatenate((input_data_array, mask_array), axis = 4)

    # pad all images to allow for an receptive field even at the borders
    padding_x, padding_y, padding_z = rf_x, rf_y, rf_z

    padded_data = np.pad(input_data_array, ((0,0), (padding_x, padding_x), (padding_y, padding_y), (padding_z, padding_z), (0,0)), mode='constant', constant_values=0)

    input_fields = rolling_window(padded_data, (0, window_d_x, window_d_y, window_d_z, 0))

    # for every voxel for each subject a receptive field is defined as the flat array of a 3D area with n_c channels
    inputs = input_fields[:, :, :, :, 0:n_c, ...].reshape((n_subjects * n_voxels_per_subject, receptive_field_size))

    if mask_array is not None:
        # obtain the number of undefined voxels for every receptive field
        cardinal_undef_in_rf = np.count_nonzero(input_fields[:, :, :, :, -1, ...] == 0, axis=(-1, -2, -3))
        # add the number of undef voxels to the input
        inputs = np.concatenate((inputs, cardinal_undef_in_rf.reshape(-1, 1)), axis=1)

    outputs = np.stack(output_data_array).reshape(n_subjects * n_voxels_per_subject)

    if not np.isnan(include_only):
        if verbose:
            print('Keeping', inputs.length / include_only.length * 100, '% of data')
        inputs = inputs[include_only]
        outputs = outputs[include_only]


    if verbose:
        print('Entire dataset. Input shape: ', inputs.shape,
              ' and output shape: ', outputs.shape)

    return inputs, outputs
