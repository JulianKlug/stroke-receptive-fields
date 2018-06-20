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

def get_undersample_selector_array(y):
    # useful for multidimensional arrays
    flat_labels = np.squeeze(y.reshape(-1, 1))
    undersampled_indices, unselected_indices = index_undersample_balance(flat_labels)
    selector = np.full(flat_labels.shape, False)
    selector[undersampled_indices] = True
    print(y.shape)
    selector = selector.reshape(y.shape)

    return selector

def index_undersample_balance(y):
    """
    Find indeces fitting a undersampled balance

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)

    Returns:
        undersampled_indices : indeces retained after random undersapling
        unselected_indices : indeces rejected after randum undersampling
    """
    print('Undesampling Ratio 1:1')
    n_pos = np.sum(y)
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
