import os
import nibabel as nib
import numpy as np
from clinical_data.clinical_data_loader import load_clinical_data

# provided a given directory return list of paths to ct_sequences and lesion_maps
def get_paths_and_ids(data_dir, ct_sequences, mri_sequences):

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    ids = []
    lesion_paths = []
    ct_paths = []
    brain_mask_paths = []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        ct_channels = []
        lesion_map = []
        brain_mask = []

        if os.path.isdir(subject_dir):
            modalities = [o for o in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir,o))]

            for modality in modalities:
                modality_dir = os.path.join(subject_dir, modality)

                studies = os.listdir(modality_dir)

                for study in studies:
                    if study.startswith(tuple(mri_sequences)):
                        lesion_map.append(os.path.join(modality_dir, study))
                    if study == 'brain_mask.nii':
                        brain_mask.append(os.path.join(modality_dir, study))
                        print('Found mask for', subject)

                # Force order specified in ct_sequences
                for channel in ct_sequences:
                    indices = [i for i, s in enumerate(studies) if channel in s]
                    if len(indices) > 1:
                        raise ValueError('Multiple images found for', channel, 'in', studies)
                    if len(indices) == 1:
                        study = studies[indices[0]]
                        ct_channels.append(os.path.join(modality_dir, study))

        if len(ct_sequences) == len(ct_channels) and len(mri_sequences) == len(lesion_map) and len(brain_mask) == 1:
            lesion_paths.append(lesion_map[0])
            brain_mask_paths.append(brain_mask[0])
            ct_paths.append(ct_channels)
            ids.append(subject)
            print('Adding', subject)
        else :
            print('Not all images found for this folder. Skipping.', subject)

    return (ids, ct_paths, lesion_paths, brain_mask_paths)

# Load nifi image maps from paths (first image is used as reference for dimensions)
# - ct_paths : list of lists of paths of channels
# - lesion_paths : list of paths of lesions maps
# - brain_mask_paths : list of paths of ct brain masks
# - return three lists containing image data for cts (as 4D array), brain masks and lesion_maps
def load_images(ct_paths, lesion_paths, brain_mask_paths):
    if len(ct_paths) != len(lesion_paths):
        raise ValueError('Number of CT and number of lesions maps should be the same.', len(ct_paths), len(lesion_paths))

    # get dimensions by extracting first image
    first_image = nib.load(ct_paths[0][0])
    first_image_data = first_image.get_data()
    n_x, n_y, n_z = first_image.shape
    n_c = len(ct_paths[0])
    print(n_c, 'channels found.')

    ct_inputs = np.empty((len(ct_paths), n_x, n_y, n_z, n_c))
    lesion_outputs = np.empty((len(lesion_paths), n_x, n_y, n_z))
    brain_masks = np.empty((len(lesion_paths), n_x, n_y, n_z), dtype = bool)

    for subject in range(len(ct_paths)):
        ct_channels = ct_paths[subject]
        for c in range(n_c):
            image = nib.load(ct_channels[c])
            image_data = image.get_data()
            if first_image_data.shape != image_data.shape:
                raise ValueError('Image does not have correct dimensions.', ct_channels[c])

            if np.isnan(image_data).any():
                print('CT images of', subject, 'contains NaN. Converting to 0.')
                image_data = np.nan_to_num(image_data)

            ct_inputs[subject, :, :, :, c] = image_data

        lesion_image = nib.load(lesion_paths[subject])
        lesion_data = lesion_image.get_data()
        # sanitize lesion data to contain only single class
        lesion_data[lesion_data > 1] = 1
        if np.isnan(lesion_data).any():
            print('Lesion label of', subject, 'contains NaN. Converting to 0.')
            lesion_data = np.nan_to_num(lesion_data)
        lesion_outputs[subject, :, :, :] = lesion_data

        brain_mask_image = nib.load(brain_mask_paths[subject])
        brain_mask_data = brain_mask_image.get_data()
        if np.isnan(brain_mask_data).any():
            print('Brain mask of', subject, 'contains NaN. Converting to 0.')
            brain_mask_data = np.nan_to_num(brain_mask_data)
        brain_masks[subject, :, :, :] = brain_mask_data

    return ct_inputs, lesion_outputs, brain_masks


def load_nifti(main_dir, ct_sequences, mri_sequences):
    ids, ct_paths, lesion_paths, brain_mask_paths = get_paths_and_ids(main_dir, ct_sequences, mri_sequences)
    return (ids, load_images(ct_paths, lesion_paths, brain_mask_paths))

# Save data as compressed numpy array
def load_and_save_data(save_dir, main_dir, clinical_dir = None, clinical_name = None, ct_sequences = [], mri_sequences = [], external_memory=False):
    """
    Load data
        - Image data (from preprocessed Nifti)
        - Clinical data (from excel)

    Args:
        save_dir : directory to save data_to
        main_dir : directory containing images
        clinical_dir (optional) : directory containing clinical data (excel)
        ct_sequences (optional, array) : array with names of ct sequences
        mri_sequences (optional, array) : array with names of mri sequences
        external_memory (optional, default False): on external memory usage, NaNs need to be converted to -1

    Returns:
        'clinical_data': numpy array containing the data for each of the patients [patient, (n_parameters)]
    """
    if len(ct_sequences) < 1:
        ct_sequences = ['wcoreg_Tmax', 'wcoreg_CBF', 'wcoreg_MTT', 'wcoreg_CBV']
        # ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_CBF', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV']
        # ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV']
        # ct_sequences = ['wcoreg_RAPID_TMax_[s]']
    if len(mri_sequences) < 1:
        # Import VOI GT with brain mask applied
        # to avoid False negatives in areas that cannot be predicted (as their are not part of the RAPID perf maps)
        mri_sequences = ['masked_wcoreg_VOI']

    print('Sequences used', ct_sequences, mri_sequences)

    included_subjects = np.array([])
    clinical_data = np.array([])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ids, (ct_inputs, lesion_GT, brain_masks) = load_nifti(main_dir, ct_sequences, mri_sequences)
    ids = np.array(ids)

    if clinical_dir is not None:
        included_subjects, clinical_data = load_clinical_data(ids, clinical_dir, clinical_name, external_memory=external_memory)

        # Remove patients with exclusion criteria
        ct_inputs = ct_inputs[included_subjects]
        lesion_GT = lesion_GT[included_subjects]

        print('Excluded', ids.shape[0] - ct_inputs.shape[0], 'subjects.')

    print('Saving a total of', ct_inputs.shape[0], 'subjects.')
    np.savez_compressed(os.path.join(save_dir, 'data_set'),
        params = {'ct_sequences': ct_sequences, 'mri_sequences': mri_sequences},
        ids = ids, included_subjects = included_subjects,
        clinical_inputs = clinical_data, ct_inputs = ct_inputs,
        lesion_GT = lesion_GT, brain_masks = brain_masks)

def load_saved_data(data_dir):
    params = np.load(os.path.join(data_dir, 'data_set.npz'))['params']
    ids = np.load(os.path.join(data_dir, 'data_set.npz'))['ids']
    clinical_inputs = np.load(os.path.join(data_dir, 'data_set.npz'))['clinical_inputs']
    ct_inputs = np.load(os.path.join(data_dir, 'data_set.npz'))['ct_inputs']
    lesion_GT = np.load(os.path.join(data_dir, 'data_set.npz'))['lesion_GT']
    brain_masks = np.load(os.path.join(data_dir, 'data_set.npz'))['brain_masks']

    print('Loading a total of', ct_inputs.shape[0], 'subjects.')
    print('Sequences used:', params)
    print(ids.shape[0] - ct_inputs.shape[0], 'subjects had been excluded.')

    return (clinical_inputs, ct_inputs, lesion_GT, brain_masks, ids, params)
