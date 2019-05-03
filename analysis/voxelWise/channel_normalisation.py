import numpy as np


def normalise_channel(all_channel_data, mask, channel):
    """
    The selected channel is normalised by dividing all its data by the median value
    in healthy tissue. Healthy tissue is defined as the tissue where Tmax < 4s.

    Args:
        all_channel_data : image input data for all subjects in form of an np array [..., c]
            Data should not be scaled beforehand!
        mask: boolean array differentiating brain from brackground [...]
        channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

    Returns: normalised_channel
    """
    channel_to_normalise = all_channel_data[... ,channel]
    Tmax = all_channel_data[..., 0]
    normalised_channel = np.empty(channel_to_normalise.shape)

    masked_healthy_image_voxels = channel_to_normalise[np.all([mask, Tmax < 4], axis = 0)]
    median_channel_healthy_tissue = np.median(masked_healthy_image_voxels)
    normalised_channel = np.divide(channel_to_normalise, median_channel_healthy_tissue)

    return normalised_channel


def multi_subj_channel_normalisation(all_input_data, masks, channel):
    """
    Channel normalisation for multiple subjects

    Args:
        all_input_data : image input data for all subjects in form of an np array [subject, ..., c]
            Data should not be scaled beforehand!
        masks: boolean array differentiating brain from brackground [subject, ...]
        channel : 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

    Returns: normalised_channel
    """
    channels =  {0:'Tmax', 1:'CBF', 2:'MTT', 3:'CBV'}
    print('Normalising channel', channels[channel])
    print('NO FEATURE SCALING SHOULD BE APPLIED TO TMAX BEFOREHAND')
    channel_to_normalise = all_input_data[:,:,:,:,channel]
    normalised_channel = np.empty(channel_to_normalise.shape)

    for p in range(channel_to_normalise.shape[0]):
        normalised_channel[p] = normalise_channel(all_input_data[p], masks[p], channel)
    return normalised_channel
