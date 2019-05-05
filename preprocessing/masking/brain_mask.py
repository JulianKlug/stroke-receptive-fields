import sys
sys.path.insert(0, '../../analysis')

import os, subprocess
import nibabel as nib
import numpy as np
import skimage.measure as measure
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation
import matplotlib.pyplot as plt

def getMask(path):
    img = nib.load(path)
    data = img.get_data()
    labeled, n_labels = measure.label(data, background = -1, return_num = True)
    labeled = labeled == 1
    labeled = labeled.astype(int)
    return labeled

def createCSFMask(data_dir, spc_name, brain_labels):
    """
    mean and standard deviation were determined of the intensity histogram within [0,400] HU.
    An upper intensity threshold μ − 1.96σ was applied to segment CSF
    followed by erosion with a structuring element with 1 voxel radius to remove potential false positive voxels due to noise,
    and connected region growing using the same thresholds.
    The CSF mask was subjected to a morphological closing operation with a structuring element with 1 voxel radius.
    """
    # METHOD 1: according to scientific reports paper
    path = os.path.join(data_dir, spc_name)
    img = nib.load(path)
    data = img.get_data()
    data[np.isnan(data)] = 0
    data[data < 3.8] = 0
    clipped_range_data = data[(data > 0) & (data < 40)]
    # clipped_range_data = data[(data > 0) & (data < 40) & (brain_labels == 1)]
    # intensity threshold μ − 1.96σ
    print(np.mean(clipped_range_data), np.std(clipped_range_data))
    CSF_threshold = np.mean(clipped_range_data) - 1.96 * np.std(clipped_range_data)
    # print(np.mean(clipped_range_data), np.std(clipped_range_data))
    CSF_mask = np.zeros(data.shape)

    print(CSF_threshold)
    CSF_mask[(data < CSF_threshold)] = 1
    # CSF_mask[(brain_labels == 1) & (data < CSF_threshold)] = 1

    CSF_mask = binary_closing(CSF_mask)

    CSF_mask = binary_erosion(CSF_mask)
    CSF_mask = binary_dilation(CSF_mask)

    # with fsl fast t1 5 class segmentation on spc
    # fsl_path = '/usr/local/fsl/bin/'
    out_dir = os.path.join(data_dir, 'fast_5seg')
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    # coordinate_space = img.affine
    # image_extension = '.nii'
    # betted = np.zeros(data.shape)
    # betted[brain_labels == 1] = data[brain_labels == 1]
    # betted_img = nib.Nifti1Image(betted, affine=coordinate_space)
    # betted_path = os.path.join(out_dir,  'betted_spc' + image_extension)
    # nib.save(betted_img, betted_path)

    # subprocess.call(['fast', '-t', '1', '-n', '5', '-H', '0.1', '-I', '4', '-l', '20.0',
    #                     '-o', os.path.join(out_dir, spc_name), spc_name], cwd = data_dir)
    # csf_class_proba_img = nib.load(os.path.join(out_dir, spc_name[:-4] + '_pve_2.nii.gz'))
    # csf_class_proba_data = csf_class_proba_img.get_data()
    # CSF_mask = np.zeros(csf_class_proba_data.shape)
    # CSF_mask[csf_class_proba_data > 0.3] = 1
    # CSF_mask = binary_erosion(CSF_mask)
    # CSF_mask = binary_dilation(CSF_mask)
    # CSF_mask = binary_closing(CSF_mask)


    return CSF_mask


def createBrainMask(data_dir, ct_sequences, spc_name, save = True):
    """
    Create a brain mask according to RAPID maps (and save it)

    Args:
        data_dir: string of path to directory in which the RAPID maps are found
        ct_sequences: list of what ct_sequences to use
        save: if mask should be saved, optional

    Returns: labeled - brain mask image
    """
    studies = os.listdir(data_dir)
    channel_masks = []
    channel_paths = []

    for study in studies:
        if study.startswith(tuple(ct_sequences)):
            channel_paths.append(os.path.join(data_dir, study))
            channel_masks.append(getMask(os.path.join(data_dir, study)))

    combined_labels = np.array(channel_masks).sum(axis=0)
    combined_labels = -1 * (combined_labels > 3) + 1

    labeled = combined_labels.astype(int)

    print(spc_name, len(channel_masks), combined_labels.shape)
    csf = createCSFMask(data_dir, spc_name, labeled)
    # labeled[csf == 1] = 0

    inverse_labeled = 1 - labeled


    if (save):
        ref_img = nib.load(channel_paths[0])
        coordinate_space = ref_img.affine
        image_extension = '.nii'
        labeled_img = nib.Nifti1Image(labeled, affine=coordinate_space)
        nib.save(labeled_img, os.path.join(data_dir,  'brain_mask' + image_extension))
        inverse_labeled_img = nib.Nifti1Image(inverse_labeled, affine=coordinate_space)
        nib.save(inverse_labeled_img, os.path.join(data_dir,  'inverse_brain_mask' + image_extension))

        csf_img = nib.Nifti1Image(csf.astype(int), affine=coordinate_space)
        nib.save(csf_img, os.path.join(data_dir,  'betted_csf_mask' + image_extension))

    return labeled, csf

def createBrainMaskWrapper(data_dir):
    spc_base = 'wbetted_SPC_301mm_Std'
    ct_sequences = ['wcoreg_Tmax', 'wcoreg_MTT', 'wcoreg_CBV', 'wcoreg_CBF']
    # ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
    # ct_sequences = ['wcoreg_RAPID_TMax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)

        if os.path.isdir(subject_dir):
            modalities = [o for o in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir,o))]

            for modality in modalities:
                if modality.startswith('Ct') or modality.startswith('pCT'):
                    modality_dir = os.path.join(subject_dir, modality)
                    spc_name = spc_base + '_' + subject + '.nii'
                    labeled, csf = createBrainMask(modality_dir, ct_sequences, spc_name)
                    print('Processed subject:', subject)
    return labeled, csf
