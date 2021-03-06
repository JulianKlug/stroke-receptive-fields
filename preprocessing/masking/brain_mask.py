import sys, argparse
sys.path.insert(0, '../../analysis')

import os
import nibabel as nib
import numpy as np
import skimage.measure as measure

def getMask(path):
    img = nib.load(path)
    data = img.get_data()
    labeled, n_labels = measure.label(data, background = -1, return_num = True)
    labeled = labeled == 1
    labeled = labeled.astype(int)
    return labeled

def createBrainMask(data_dir, ct_sequences, csf_image_name, save = True, save_name = 'brain_mask'):
    """
    Create a brain mask (according to input maps (and save it))

    Args:
        data_dir: string of path to directory in which the RAPID maps are found
        ct_sequences: list of what ct_sequences to use
        save: if mask should be saved, optional
        save_name: name to save as

    Returns: labeled - brain mask image
    """
    csf_img = nib.load(os.path.join(data_dir, csf_image_name))
    csf_data = csf_img.get_data()
    csf_data = np.nan_to_num(csf_data, nan=1)
    csf_label = np.zeros(csf_data.shape)
    csf_label[csf_data > 0.5] = 1

    if ct_sequences:
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
    else:
        labeled = np.ones(csf_data.shape)



    # substract csf_label to brain label
    labeled[csf_label == 1] = 0

    inverse_labeled = 1 - labeled


    if (save):
        # ref_img = nib.load(channel_paths[0])
        coordinate_space = csf_img.affine
        image_extension = '.nii'
        labeled_img = nib.Nifti1Image(labeled, affine=coordinate_space)
        nib.save(labeled_img, os.path.join(data_dir,  save_name + image_extension))
        inverse_labeled_img = nib.Nifti1Image(inverse_labeled, affine=coordinate_space)
        nib.save(inverse_labeled_img, os.path.join(data_dir,  'inverse_' + save_name + image_extension))
    return labeled, csf_label

def createBrainMaskWrapper(data_dir, restrict_to_RAPID_maps=False, high_resolution=False):
    csf_image_name = 'wreor_CSF_mask.nii'
    ct_sequences = ['wcoreg_Tmax', 'wcoreg_MTT', 'wcoreg_CBV', 'wcoreg_CBF']
    save_name = 'brain_mask'
    # ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
    # ct_sequences = ['wcoreg_RAPID_TMax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']

    if high_resolution:
        csf_image_name = 'hd_CSF_mask.nii'
        ct_sequences = ['coreg_Tmax', 'coreg_MTT', 'coreg_CBV', 'coreg_CBF']
        save_name = 'hd_brain_mask'

    # do not restrict to the RAPID maps in this case (may cut off some parts)
    if not restrict_to_RAPID_maps:
        ct_sequences = []

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
                    createBrainMask(modality_dir, ct_sequences, csf_image_name, save_name = save_name)
                    print('Processed subject:', subject)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain Mask creation')
    parser.add_argument('input_directory')
    parser.add_argument("--restrict2Rapid", nargs='?', const=True, default=False, help="Mask parts outside the RAPID maps are discared.")
    parser.add_argument("--hd", nargs='?', const=True, default=False, help="Use HD version of the files")
    args = parser.parse_args()
    createBrainMaskWrapper(args.input_directory, restrict_to_RAPID_maps=args.restrict2Rapid, high_resolution=args.hd)
