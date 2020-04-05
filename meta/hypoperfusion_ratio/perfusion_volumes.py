import os, sys, argparse
sys.path.insert(0, '../../')
import pandas as pd
import numpy as np
from analysis import data_loader
from scipy.ndimage.morphology import binary_closing, binary_opening
from skimage.morphology import remove_small_objects, ball
from analysis.voxelwise.channel_normalisation import normalise_channel_by_Tmax4, normalise_channel_by_contralateral, normalise_by_contralateral_region
from analysis.utils import rescale_outliers
from analysis.dataset_visualization import visualize_dataset

def threshold_volume(data, threshold):
    tresholded_voxels = np.zeros(data.shape)
    tresholded_voxels[data > threshold] = 1
    return tresholded_voxels

def apply_csf_masks(data, masks):
    masked_images = data
    masked_images[masks < 1] = 0
    return masked_images

def smooth(data):
    smoothed_data = np.zeros(data.shape)
    for subj in range(data.shape[0]):
        smoothed_data[subj] = binary_closing(np.squeeze(data[subj]), ball(1))
        smoothed_data[subj] = binary_opening(smoothed_data[subj], ball(1))
        smoothed_data[subj] = remove_small_objects(smoothed_data[subj].astype(bool), min_size=1000, connectivity=1).astype(int)

    return smoothed_data


def compute_core(data, brain_masks, threshold=0.3, use_Tmax4_normalisation=True, use_region_wise_normalisation=True):
    '''
    Compute voxelwise ischemic core
    '''
    penumbra = threshold_volume(data[..., 0], 6)

    CBF_normalised_byTmax4, _ = normalise_channel_by_Tmax4(data[brain_masks].reshape(-1, 4), brain_masks, 1)
    CBF_normalised_byContralateral, _ = normalise_channel_by_contralateral(data[brain_masks].reshape(-1, 4), brain_masks, 1)
    CBF_normalised_byContralateral_Region = normalise_by_contralateral_region(data, three_D_region=False)[..., 1] * brain_masks

    tresholded_voxels = np.zeros(data[..., 0].shape)
    # Parts of penumbra (Tmax > 6) where CBF < 30% of healthy tissue (contralateral or region where Tmax < 4s)
    tresholded_voxels[(CBF_normalised_byContralateral < threshold) & (penumbra == 1)] = 1
    if use_Tmax4_normalisation:
        tresholded_voxels[(CBF_normalised_byTmax4 < threshold) & (penumbra == 1)] = 1
    if use_region_wise_normalisation:
        tresholded_voxels[(CBF_normalised_byContralateral_Region < threshold) & (penumbra == 1)] = 1

    return tresholded_voxels

def estimate_perfusion_volumes(data_dir):
    '''
    # compute volumes for Tmax > 10; Tmax > 6; core
    :param data_dir: directory in which to find this data
    :return:
    - an excel file with respective volumes in mm3
    - an visualisation of the outputs of the different thresholds
    '''

    data_set = data_loader.load_saved_data(data_dir, filename='data_set.npz') # do not use standardized data
    clinical_inputs, ct_inputs, ct_label, _, _, brain_masks, ids, params = data_set

    # check for scaling errors in RAPID processing
    ct_inputs = rescale_outliers(ct_inputs, MASKS=brain_masks)

    ct_inputs = apply_csf_masks(ct_inputs, brain_masks)
    if not ct_inputs.shape[4] == 4:
        raise Exception('All perfusion channels needed: Tmax, MTT, CBV, CBF')

    tmax_over_6 = threshold_volume(ct_inputs[..., 0], threshold=6)
    smooth_tmax6 = smooth(tmax_over_6)

    tmax_over_10 = threshold_volume(ct_inputs[..., 0], threshold=10)
    smooth_tmax10 = smooth(tmax_over_10)

    norm_cbf_under_30_withTmax4 = compute_core(ct_inputs, brain_masks, use_Tmax4_normalisation=True,
                                               use_region_wise_normalisation=False)
    smooth_cbf30_T4 = smooth(norm_cbf_under_30_withTmax4)

    norm_cbf_under_30_withoutTmax4 = compute_core(ct_inputs, brain_masks, use_Tmax4_normalisation=False,
                                                  use_region_wise_normalisation=False)
    smooth_cbf30_noT4 = smooth(norm_cbf_under_30_withoutTmax4)

    norm_cbf_under_30_withTmax4_and_regional = compute_core(ct_inputs, brain_masks, use_Tmax4_normalisation=True,
                                                            use_region_wise_normalisation=True)
    smooth_cbf30_T4_and_region = smooth(norm_cbf_under_30_withTmax4_and_regional)


    # Compute volumes
    tmax_over_6_volume = np.sum(tmax_over_6, axis=(1, 2, 3))
    smooth_tmax6_volume = np.sum(smooth_tmax6, axis=(1, 2, 3))

    tmax_over_10_volume = np.sum(tmax_over_10, axis=(1, 2, 3))
    smooth_tmax10_volume = np.sum(smooth_tmax10, axis=(1, 2, 3))

    norm_cbf_under_30_withTmax4_volume = np.sum(norm_cbf_under_30_withTmax4, axis=(1, 2, 3))
    smooth_cbf30_T4_volume = np.sum(smooth_cbf30_T4, axis=(1, 2, 3))

    norm_cbf_under_30_withoutTmax4_volume = np.sum(norm_cbf_under_30_withoutTmax4, axis=(1, 2, 3))
    smooth_cbf30_noT4_volume = np.sum(smooth_cbf30_noT4, axis=(1, 2, 3))

    norm_cbf_under_30_withTmax4_and_regional_volume = np.sum(norm_cbf_under_30_withTmax4_and_regional, axis=(1, 2, 3))
    smooth_cbf30_T4_and_region_volume = np.sum(smooth_cbf30_T4_and_region, axis=(1, 2, 3))

    gt_volume = np.sum(ct_label, axis=(1, 2, 3))

    # Record data
    perfusion_volumes = pd.DataFrame()
    perfusion_volumes['subject_id'] = ids
    perfusion_volumes['tmax_over_6s'] = tmax_over_6_volume
    perfusion_volumes['smooth_tmax_over_6s'] = smooth_tmax6_volume
    perfusion_volumes['tmax_over_10s'] = tmax_over_10_volume
    perfusion_volumes['smooth_tmax_over_10s'] = smooth_tmax10_volume
    perfusion_volumes['norm1_cbf_under_30p'] = norm_cbf_under_30_withoutTmax4_volume
    perfusion_volumes['smooth_norm1_cbf_under_30p'] = smooth_cbf30_noT4_volume
    perfusion_volumes['norm2_cbf_under_30p'] = norm_cbf_under_30_withTmax4_volume
    perfusion_volumes['smooth_norm2_cbf_under_30p'] = smooth_cbf30_T4_volume
    perfusion_volumes['norm3_cbf_under_30p'] = norm_cbf_under_30_withTmax4_and_regional_volume
    perfusion_volumes['smooth_norm3_cbf_under_30p'] = smooth_cbf30_T4_and_region_volume

    perfusion_volumes['ground_truth'] = gt_volume

    # Convert units: voxel to mm3
    # Formula: volumes_in_mm3 = volumes_in_vx * 8
    perfusion_volumes = perfusion_volumes.apply(lambda x: np.multiply(x, 8) if x.name != 'subject_id' else x)
    # convert to ml
    perfusion_volumes_ml = perfusion_volumes.apply(lambda x: np.multiply(x, 0.001) if x.name != 'subject_id' else x)

    # Visualisation
    stacked_volumes = np.stack((
        tmax_over_6, smooth_tmax6,
        tmax_over_10, smooth_tmax10,
        norm_cbf_under_30_withoutTmax4, smooth_cbf30_noT4,
        norm_cbf_under_30_withTmax4, smooth_cbf30_T4,
        norm_cbf_under_30_withTmax4_and_regional, smooth_cbf30_T4_and_region,
        ct_label
    ), axis=-1)
    stacked_volume_labels = [
        'tmax_over_6s', 'smooth_tmax_over_6s',
        'tmax_over_10s', 'smooth_tmax_over_10s',
        'norm1_cbf_under_30p', 'smooth_norm1_cbf_under_30p',
        'norm2_cbf_under_30p', 'smooth_norm2_cbf_under_30p',
        'norm3_cbf_under_30p', 'smooth_norm3_cbf_under_30p',
        'ground_truth'
    ]

    visualize_dataset(stacked_volumes, stacked_volume_labels, ids, data_dir, save_name='perfusion_volumes_visualisation')

    # Save data
    with pd.ExcelWriter(os.path.join(data_dir, 'perfusion_volumes.xlsx')) as writer:
        perfusion_volumes_ml.to_excel(writer, sheet_name='volumes_ml')
        perfusion_volumes.to_excel(writer, sheet_name='volumes_mm3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute perfusion volumes')
    parser.add_argument('input_directory')
    args = parser.parse_args()
    estimate_perfusion_volumes(args.input_directory)

