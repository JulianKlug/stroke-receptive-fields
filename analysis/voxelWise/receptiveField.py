import sys
sys.path.insert(0, '../')

from rolling_window import rolling_window
import numpy as np

def reshape_to_receptive_field(input_data_array, output_data_array, receptive_field_dimensions, verbose = False) :
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1

    if verbose:
        print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    n_subjects = input_data_array.shape[0]
    n_receptive_fields = input_data_array[0, :, :, :, 0].size * n_subjects  # ie voxels per image times number of images
    receptive_field_size = window_d_x * window_d_y * window_d_z * input_data_array[0,0,0,0,:].size
    n_x, n_y, n_z, n_c = input_data_array[0].shape
    n_voxels_per_subject = n_x * n_y * n_z

    # pad all images to allow for an receptive field even at the borders
    padding = max([rf_x, rf_y, rf_z])
    padded_data = np.pad(input_data_array, ((0,0), (padding, padding), (padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

    input_fields = rolling_window(padded_data, (0, window_d_x, window_d_y, window_d_z, 0))

    inputs = input_fields.reshape((n_subjects * n_voxels_per_subject, receptive_field_size))

    outputs = np.stack(output_data_array).reshape(n_subjects * n_voxels_per_subject)

    if verbose:
        print('Entire dataset. Input shape: ', inputs.shape,
              ' and output shape: ', outputs.shape)

    return inputs, outputs

def predict(input_data, model, receptive_field_dimensions, verbose = False):
    # Dimensions of the receptive field defined as distance to center point in every direction
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z = 2 * np.array(receptive_field_dimensions) + 1
    if verbose :
        print('Receptive field window dimensions are: ', window_d_x, window_d_y, window_d_z )

    n_x, n_y, n_z, n_c = input_data.shape
    if verbose:
        print('Predicting from input: ', input_data.shape)
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c
    n_voxels_per_subject = n_x * n_y * n_z

    output = np.zeros([n_x, n_y, n_z])

    # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
    padding = max([rf_x, rf_y, rf_z])
    padded_input_data = np.pad(input_data, ((padding, padding), (padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

    input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))

    inputs = input_fields.reshape((n_voxels_per_subject, receptive_field_size))

    output = model.predict_proba(inputs)
    output = output[:, 1].reshape(n_x, n_y, n_z)

    return output
