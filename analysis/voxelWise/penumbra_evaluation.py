import numpy as np
from sklearn.metrics import precision_score


def threshold_Tmax6(data):
    # Get penumbra mask
    Tmax = data[..., 0]
    tresholded_voxels = np.zeros(Tmax.shape)
    # penumbra (Tmax > 6) without extremes
    tresholded_voxels[(Tmax > 6) & (Tmax < np.percentile(Tmax, 99))] = 1 # define penumbra
    return np.squeeze(tresholded_voxels)


def penumbra_match(probas_, data):
    '''
    Get statistics comparing output (defined by output > 0.5) and penumbra as defined by Tmax > 6
    :param probas_: probability of infarction for every data point (i)
    :param data: input data for every data point (i, c)
    :return:
    '''
    penumbra = threshold_Tmax6(data) == 1

    threshold = 0.5

    # Positive predictive value : tp / (tp + fp)
    # ie ratio of prediction that is in penumbra
    ratio_in_penumbra = precision_score(penumbra, probas_[:] >= threshold)

    return ratio_in_penumbra
