import sys
sys.path.insert(0, '../')

import os
from rolling_window import rolling_window
import numpy as np
import xgboost as xgb
from ext_mem_utils import save_to_svmlight


def reshape_to_receptive_field(input_data_array, output_data_array, receptive_field_dimensions, include_only = np.NaN, verbose = False) :
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

    if not np.isnan(include_only):
        if verbose:
            print('Keeping', inputs.length / include_only.length * 100, '% of data')
        inputs = inputs[include_only]
        outputs = outputs[include_only]


    if verbose:
        print('Entire dataset. Input shape: ', inputs.shape,
              ' and output shape: ', outputs.shape)

    return inputs, outputs

def predict(input_data, test_data_path, model, receptive_field_dimensions, verbose = False, external_memory = False):
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

    output = np.zeros([n_x, n_y, n_z]).reshape((n_voxels_per_subject, 1))

    # Pad input image with 0 (neutral) border to be able to get an receptive field at corner voxels
    padding = max([rf_x, rf_y, rf_z])
    padded_input_data = np.pad(input_data, ((padding, padding), (padding, padding), (padding, padding), (0,0)), mode='constant', constant_values=0)

    input_fields = rolling_window(padded_input_data, (window_d_x, window_d_y, window_d_z, 0))

    inputs = input_fields.reshape((n_voxels_per_subject, receptive_field_size))

    if not external_memory:
        dtest = xgb.DMatrix(inputs)
        output = model.predict(dtest)
        output = 1 - output.reshape(n_x, n_y, n_z)


        # output = model.predict_proba(inputs)
        # output = 1 - output[:, 1].reshape(n_x, n_y, n_z)

    else:
        test_data_path = os.path.join(test_data_path, 'testData.txt')
        print(inputs.shape, output.shape)
        save_to_svmlight(inputs, output, test_data_path)

        data = xgb.DMatrix(test_data_path)
        print(data.get_label().shape)
        output = model.predict(data, ntree_limit=0)
        output = 1 - output[0:n_voxels_per_subject].reshape(n_x, n_y, n_z)
        os.remove(test_data_path) # Clean-up

    return output
