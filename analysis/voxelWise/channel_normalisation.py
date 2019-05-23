import numpy as np
from dimension_utils import reconstruct_image


def normalise_channel_by_Tmax4(all_channel_data, data_positions, channel):
    """
    The selected channel is normalised by dividing all its data by the median value
    in healthy tissue. Healthy tissue is defined as the tissue where Tmax < 4s.

    Args:
        all_channel_data : image input data for all subjects in form of an np array [..., c]
            Data should not be scaled beforehand!
        data_positions: boolean map in original space with 1 as placeholder for data points
        channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV

    Returns: normalised_channel
    """
    # reconstruct to be able to separate individual subjects
    reconstructed = reconstruct_image(all_channel_data, data_positions, all_channel_data.shape[-1])
    channel_to_normalise = reconstructed[..., channel]
    Tmax = reconstructed[..., 0]
    normalised_channel = np.zeros(channel_to_normalise.shape)
    for subj in range(data_positions.shape[0]):
        # todo detect cases where Tmax is scaled x10
        masked_healthy_image_voxels = channel_to_normalise[subj][np.all([data_positions[subj] == 1, Tmax[subj] < 4], axis=0)]
        # clip to remove extremes that falsify results
        masked_healthy_image_voxels = np.clip(
            masked_healthy_image_voxels, np.percentile(masked_healthy_image_voxels, 1),
            np.percentile(masked_healthy_image_voxels, 99))
        median_channel_healthy_tissue = np.median(masked_healthy_image_voxels)
        normalised_channel[subj] = np.divide(channel_to_normalise[subj], median_channel_healthy_tissue)

    image_normalised_channel = normalised_channel
    flat_normalised_channel = normalised_channel[data_positions == 1].reshape(-1)
    return image_normalised_channel, flat_normalised_channel

def normalise_channel_by_contralateral(all_channel_data, data_positions, channel):
    '''
    Normalise a channel by dividing every voxel by the median value of contralateral side
    :param all_channel_data: image input data for all subjects in form of an np array [i, c]
    :param data_positions: boolean map in original space with 1 as placeholder for data points
    :param channel: channel to normalise: 0 - Tmax, 1 - CBF, 2 - MTT, 3 - CBV
    :return: (image_normalised_channel, flat_normalised_channel) - normalised channel in 3D and flat form
    '''
    # Recover 3D shape of data
    reconstructed_data = reconstruct_image(all_channel_data, data_positions, all_channel_data.shape[-1])
    channel_to_normalise_data = reconstructed_data[... ,channel]
    normalised_channel = np.zeros(channel_to_normalise_data.shape, dtype=np.float64)
    x_center = channel_to_normalise_data.shape[1] // 2

    for subj in range(channel_to_normalise_data.shape[0]):
        # normalise left side
        right_side = channel_to_normalise_data[subj][x_center:][data_positions[subj][x_center:] == 1]
        clipped_right_side = np.clip(right_side, np.percentile(right_side, 1), np.percentile(right_side, 99))
        right_side_median = np.nanmedian(clipped_right_side)
        normalised_channel[subj][:x_center] = np.divide(channel_to_normalise_data[subj][:x_center], right_side_median)

        # normalise right side
        left_side = channel_to_normalise_data[subj][:x_center][data_positions[subj][:x_center] == 1]
        clipped_left_side = np.clip(left_side, np.percentile(left_side, 1), np.percentile(left_side, 99))
        left_side_median = np.nanmedian(clipped_left_side)
        normalised_channel[subj][x_center:] = np.divide(channel_to_normalise_data[subj][x_center:], left_side_median)

    image_normalised_channel = normalised_channel
    flat_normalised_channel = normalised_channel[data_positions == 1].reshape(-1)
    return image_normalised_channel, flat_normalised_channel


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
        normalised_channel[p] = normalise_channel_by_contralateral(all_input_data[p][masks[p]], masks[p], channel)
    return normalised_channel
