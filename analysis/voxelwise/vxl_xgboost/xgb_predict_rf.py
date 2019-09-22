import sys, os
sys.path.insert(0, '../')

from rolling_window import rolling_window
import numpy as np
import xgboost as xgb
from voxelwise.ext_mem_utils import save_to_svmlight

def xgb_predict(input_data, test_data_path, model, receptive_field_dimensions, verbose = False, external_memory = False):
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
        # output = 1 - output[0:n_voxels_per_subject].reshape(n_x, n_y, n_z)
        output = output[0:n_voxels_per_subject].reshape(n_x, n_y, n_z)
        os.remove(test_data_path) # Clean-up

    return output
