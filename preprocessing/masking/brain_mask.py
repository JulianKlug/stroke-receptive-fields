import sys
sys.path.insert(0, '../../analysis')

import os
import nibabel as nib
import numpy as np
import skimage.measure as measure
import matplotlib.pyplot as plt

def getMask(path):
    img = nib.load(path)
    data = img.get_data()
    labeled, n_labels = measure.label(data, background = -1, return_num = True)
    labeled = labeled == 1
    labeled = labeled.astype(int)
    return labeled


def createBrainMask(data_dir, ct_sequences, save = True):
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

    if (save):
        ref_img = nib.load(channel_paths[0])
        coordinate_space = ref_img.affine
        image_extension = '.nii'
        labeled_img = nib.Nifti1Image(labeled, affine=coordinate_space)
        nib.save(labeled_img, os.path.join(data_dir,  'brain_mask' + image_extension))

    return labeled

def createBrainMaskWrapper(data_dir):
    ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)

        if os.path.isdir(subject_dir):
            modalities = [o for o in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir,o))]

            for modality in modalities:
                if (modality.startswith('Ct')):
                    modality_dir = os.path.join(subject_dir, modality)
                    createBrainMask(modality_dir, ct_sequences)
                    print('Processed subject:', subject)


dir = '/home/klug/data/preprocessed_original_masked'
createBrainMaskWrapper(dir)
