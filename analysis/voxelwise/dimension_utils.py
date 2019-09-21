import numpy as np

def reconstruct_image(data_points, data_positions, n_channels):
    '''
    Reconstruct 2D/3D shape of images with available data filled in (may not be whole image because of sampling)
    :param data_points: data for every available voxel [i, c]
    :param data_positions: (n, x, y, z) boolean array where True marks a given data_point
    :return: reconstructed
    '''
    reconstructed = np.zeros(data_positions.shape + tuple([n_channels]))
    reconstructed[data_positions == 1] = data_points
    return reconstructed