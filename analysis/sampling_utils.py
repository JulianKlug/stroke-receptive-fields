import sys
sys.path.insert(0, './voxelWise')

import numpy as np
from collections import Counter
from ext_mem_utils import delete_lines
import xgboost as xgb

def undersample_by_index(X, y):
    print('initial', X.shape, y.shape)
    undersampled_indices, unselected_indices = index_undersample_balance(y)
    return (X[undersampled_indices], y[undersampled_indices])
    # return (np.delete(X, unselected_indices, 0), np.delete(y, unselected_indices, 0))

def ext_mem_undersample(datapath):
    print('Undersampling data:', datapath)
    data = xgb.DMatrix(datapath)
    labels = data.get_label()
    undersampled_indices, unselected_indices = index_undersample_balance(labels)
    print('unselect lines', unselected_indices.shape)
    delete_lines(datapath, unselected_indices)

def get_undersample_selector_array(y, mask = None):
    """
    Return boolean array with true for indeces fitting a undersampled balance
    Useful for multidimensional arrays

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)
        mask : mask representing the areas where negative samples of y can be taken from

    Returns:
        selector : boolean with true indeces retained after random undersapling
    """
    flat_labels = np.squeeze(y.reshape(-1, 1))
    flat_mask = None
    if mask is not None:
        print('Using a mask for sampling.')
        flat_mask = np.squeeze(mask.reshape(-1, 1))
    undersampled_indices, unselected_indices = index_undersample_balance(flat_labels, flat_mask)
    selector = np.full(flat_labels.shape, False)
    selector[undersampled_indices] = True
    selector = selector.reshape(y.shape)

    return selector

def index_undersample_balance(y, mask = None):
    """
    Find indeces fitting a undersampled balance

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)
        mask : mask representing the areas where negative samples of y can be taken from

    Returns:
        undersampled_indices : indeces retained after random undersapling
        unselected_indices : indeces rejected after random undersampling
    """
    print('Undersampling Ratio 1:1')
    n_pos = np.sum(y)
    if mask is not None:
        # Only take negatives out of the mask (positives are always in the mask)
        masked_negatives = np.all([y == 0, mask == 1], axis = 0)
        neg_indices = np.squeeze(np.argwhere(masked_negatives))
    else :
        neg_indices = np.squeeze(np.argwhere(y == 0))
    pos_indices = np.squeeze(np.argwhere(y == 1))
    randomly_downsampled_neg_indices = np.random.choice(neg_indices, int(n_pos), replace = False)
    undersampled_indices = np.concatenate([pos_indices, randomly_downsampled_neg_indices])
    unselected_indices = np.setdiff1d(neg_indices, randomly_downsampled_neg_indices)

    return (undersampled_indices, unselected_indices)

def balance(X, y, verbose = False):
    # Prefer under_sampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)

    # Avoid over sampling because it overloads the RAM
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, y)

    if (verbose):
        print('Balancing Data.')
        print('Remaining data points after balancing: ', sorted(Counter(y_resampled).items()))

    return (X_resampled, y_resampled)

def get_controlateral_side_mask(self, all_data, VOI):
    '''
    Get mask of the controlateral side of the space
    :param all_data: all image data with shape [x, y, z, c]
    :param VOIs: coordinates of region of which the contralateral side is defined [x, y, z]
    :return: controlateral_side_mask, mask marking contralateral side of VOI in image as True
    '''
    x_center = all_data.shape[0] // 2
    controlateral_side_mask = np.full(all_data.shape[:4], False)
    # VOI is on the right
    if VOI[0] > x_center:
        # return left side
        controlateral_side_mask[:x_center, :, :] = True
    else: # VOI is on the left
        # return right side
        controlateral_side_mask[x_center:, :, :] = True
    return controlateral_side_mask