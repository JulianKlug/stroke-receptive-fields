import numpy as np

def normalise_channel(all_input_data, masks, channel):
    """
    The selected channel is normalised by dividing all its data by the median value
    in healthy tissue. Healthy tissue is defined as the tissue where Tmax < 4s.

    Args:
        all_input_data : image input data for all subjects in form of an np array [subject, x, y, z, c]
            Data should not be scaled beforehand!
        masks: boolean array differentiating brain from brackground [subject, x, y, z]
        channel : 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

    Returns: normalised_channel
    """
    channels =  {0:'Tmax', 1:'CBF', 2:'MTT', 3:'CBV'}
    print('Normalising channel', channels[channel])
    print('NO FEATURE SCALING SHOULD BE APPLIED TO TMAX BEFOREHAND')
    channel_to_normalise = all_input_data[:,:,:,:,channel]
    Tmax = all_input_data[:,:,:,:,0]
    normalised_channel = np.empty(channel_to_normalise.shape)

    for p in range(channel_to_normalise.shape[0]):
        masked_healty_image_voxels = channel_to_normalise[p,:,:,:][np.all([masks[p], Tmax[p,:,:,:] < 4], axis = 0)]
        median_channel_healthy_tissue = np.median(masked_healty_image_voxels)
        normalised_channel[p,:,:,:] = np.divide(channel_to_normalise[p,:,:,:], median_channel_healthy_tissue)
    return normalised_channel
