from cv2 import GaussianBlur
import numpy as np

def gaussian_smoothing(data, kernel_shape = (5, 5)):
    '''
    Smooth a set of n images with a 2D gaussian kernel on their x, y planes iterating through z
    Every plane in z is smoothed independently
    Every channel is smoothed independently
    :param data: images to smooth (n, x, y, z, c)
    :param kernel_shape: 2D kernel shape (w, h)
        Default is (5, 5) - (10mm, 10mm), 5mm radius as inspired by
        Campbell Bruce C.V., Christensen Søren, Levi Christopher R., Desmond Patricia M., Donnan Geoffrey A., Davis Stephen M., et al. Cerebral Blood Flow Is the Optimal CT Perfusion Parameter for Assessing Infarct Core. Stroke. 2011 Dec 1;42(12):3435–40.
    :return: smoothed_data
    '''
    smoothed_data = np.empty(data.shape)
    if len(data.shape) != 5:
        raise ValueError('Shape of data to smooth should be (n, x, y, z, c) and not', data.shape)
    for i in range(data.shape[0]):
        for c in range(data.shape[4]):
            smoothed_data[i, :, :, :, c] = GaussianBlur(data[i, :, :, :, c], kernel_shape, 0)
    return smoothed_data
