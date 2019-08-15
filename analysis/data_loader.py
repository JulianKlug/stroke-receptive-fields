import os
import nibabel as nib
import numpy as np
from clinical_data.clinical_data_loader import load_clinical_data
from utils import find_max_shape

# provided a given directory return list of paths to ct_sequences and lesion_maps
def get_paths_and_ids(data_dir, ct_sequences, ct_label_sequences, mri_sequences, mri_label_sequences, brain_mask_name):

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    ids = []
    ct_lesion_paths = []
    ct_paths = []
    mri_lesion_paths = []
    mri_paths = []
    brain_mask_paths = []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        ct_channels = []
        ct_lesion_map = []
        mri_channels = []
        mri_lesion_map = []
        brain_mask = []

        if os.path.isdir(subject_dir):
            modalities = [o for o in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir,o))]

            for modality in modalities:
                modality_dir = os.path.join(subject_dir, modality)

                studies = os.listdir(modality_dir)

                for study in studies:
                    if study.startswith(tuple(ct_label_sequences)):
                        ct_lesion_map.append(os.path.join(modality_dir, study))
                    if study.startswith(tuple(mri_label_sequences)):
                        mri_lesion_map.append(os.path.join(modality_dir, study))
                    if study == brain_mask_name:
                        brain_mask.append(os.path.join(modality_dir, study))
                        print('Found mask for', subject)

                # Force order specified in ct_sequences
                for channel in ct_sequences:
                    indices = [i for i, s in enumerate(studies) if s.startswith(channel)]
                    if len(indices) > 1:
                        raise ValueError('Multiple images found for', channel, 'in', studies)
                    if len(indices) == 1:
                        study = studies[indices[0]]
                        ct_channels.append(os.path.join(modality_dir, study))

                # force order specified in mri_sequences
                for sequence in mri_sequences:
                    indices = [i for i, s in enumerate(studies) if s.startswith(sequence)]
                    if len(indices) == 0: continue
                    if len(indices) == 1:
                        study = studies[indices[0]]
                        mri_channels.append(os.path.join(modality_dir, study))
                        continue
                    if 'TRACE' in sequence:
                        trace_studies = [studies[i] for i in indices]
                        trace_studies = sorted(trace_studies, key=str.lower)
                        # only choose the first 2 TRACE channels
                        mri_channels.append(os.path.join(modality_dir, trace_studies[0]))
                        mri_channels.append(os.path.join(modality_dir, trace_studies[1]))
                        continue
                    raise ValueError('Multiple images found for', sequence, 'in', studies, [studies[i] for i in indices])

        # as there are 2 TRACE sequences
        n_mri_sequences = len(mri_sequences)
        if np.any(['TRACE' in sequence for sequence in mri_sequences]): n_mri_sequences += 1

        if len(ct_sequences) == len(ct_channels) and len(ct_label_sequences) == len(ct_lesion_map) \
                and n_mri_sequences == len(mri_channels) and len(mri_label_sequences) == len(mri_lesion_map) \
                and len(brain_mask) == 1:
            ct_lesion_paths.append(ct_lesion_map[0])
            if len(mri_lesion_map) > 0:
                mri_lesion_paths.append(mri_lesion_map[0])
            brain_mask_paths.append(brain_mask[0])
            ct_paths.append(ct_channels)
            mri_paths.append(mri_channels)
            ids.append(subject)
            print('Adding', subject)
        else :
            print('Not all images found for this folder. Skipping.', subject)

    return (ids, ct_paths, ct_lesion_paths, mri_paths, mri_lesion_paths, brain_mask_paths)

# Load nifi image maps from paths (first image is used as reference for dimensions)
# - ct_paths : list of lists of paths of channels
# - lesion_paths : list of paths of lesions maps
# - brain_mask_paths : list of paths of ct brain masks
# - return three lists containing image data for cts (as 4D array), brain masks and lesion_maps
def load_images(ct_paths, ct_lesion_paths, mri_paths, mri_lesion_paths, brain_mask_paths, ids, high_resolution = False):
    if len(ct_paths) != len(ct_lesion_paths):
        raise ValueError('Number of CT and number of lesions maps should be the same.', len(ct_paths), len(ct_lesion_paths))

    # get CT dimensions by extracting first image
    first_image = nib.load(ct_paths[0][0])
    first_image_data = first_image.get_data()
    n_x, n_y, n_z = first_image.shape
    ct_n_c = len(ct_paths[0])
    print(ct_n_c, 'CT channels found.')

    ct_inputs = np.empty((len(ct_paths), n_x, n_y, n_z, ct_n_c))
    ct_lesion_outputs = np.empty((len(ct_lesion_paths), n_x, n_y, n_z))
    brain_masks = np.empty((len(ct_lesion_paths), n_x, n_y, n_z), dtype = bool)

    # get MRI dimensions by extracting first image
    if mri_paths[0]:
        mri_first_image = nib.load(mri_paths[0][0])
        mri_first_image_data = mri_first_image.get_data()
        mri_n_x, mri_n_y, mri_n_z = mri_first_image_data.shape
        mri_n_c = len(mri_paths[0])
        print(mri_n_c, 'MRI channels found.')

        mri_inputs = np.empty((len(mri_paths), mri_n_x, mri_n_y, mri_n_z, mri_n_c))
        mri_lesion_outputs = np.empty((len(mri_lesion_paths), mri_n_x, mri_n_y, mri_n_z))

    # for high res images not all images have the same shape
    if high_resolution:
        data_dir = '/'.join(ct_paths[0][0].split('/')[:-3])
        max_shape_x, max_shape_y, max_shape_z = find_max_shape(data_dir, 'coreg_Tmax')
        ct_inputs = np.empty((len(ct_paths), max_shape_x, max_shape_y, max_shape_z, ct_n_c))
        ct_lesion_outputs = np.empty((len(ct_lesion_paths), max_shape_x, max_shape_y, max_shape_z))
        brain_masks = np.empty((len(ct_lesion_paths), max_shape_x, max_shape_y, max_shape_z), dtype=bool)

        if mri_paths[0]:
            mri_inputs = np.empty((len(mri_paths), max_shape_x, max_shape_y, max_shape_z))
            mri_lesion_outputs = np.empty((len(mri_lesion_paths), max_shape_x, max_shape_y, max_shape_z, mri_n_c))

    def rectify_shape(image_data):
        # high resolution data has to be padded to have constant dimensions
        if high_resolution and image_data.shape[2] < max_shape_z:
            n_missing_layers = max_shape_z - image_data.shape[2]
            image_data = np.pad(image_data, ((0, 0), (0, 0), (n_missing_layers, 0)), 'constant', constant_values=0)
        return image_data

    for subject in range(len(ct_paths)):
        ct_channels = ct_paths[subject]
        for c in range(ct_n_c):
            image = nib.load(ct_channels[c])
            image_data = image.get_data()
            image_data = rectify_shape(image_data)

            if ct_inputs[subject, :, :, :, c].shape != image_data.shape:
                raise ValueError('Image does not have correct dimensions.', ct_channels[c])

            if np.isnan(image_data).any():
                print('CT images of', ids[subject], 'contains NaN. Converting to 0.')
                image_data = np.nan_to_num(image_data)

            ct_inputs[subject, :, :, :, c] = image_data

        # load mri channels
        if mri_paths[0]:
            mri_channels = mri_paths[subject]
            for k in range(mri_n_c):
                image = nib.load(mri_channels[k])
                image_data = image.get_data()
                image_data = rectify_shape(image_data)

                if mri_inputs[subject, :, :, :, k].shape != image_data.shape:
                    raise ValueError('Image does not have correct dimensions.', mri_channels[k])

                if np.isnan(image_data).any():
                    print('MRI images of', ids[subject], 'contains NaN. Converting to 0.')
                    image_data = np.nan_to_num(image_data)

                mri_inputs[subject, :, :, :, k] = image_data

        # load ct labels
        ct_lesion_image = nib.load(ct_lesion_paths[subject])
        ct_lesion_data = ct_lesion_image.get_data()
        ct_lesion_data = rectify_shape(ct_lesion_data)

        # load MRI labels
        if mri_lesion_paths:
            mri_lesion_image = nib.load(mri_lesion_paths[subject])
            mri_lesion_data = mri_lesion_image.get_data()
            mri_lesion_data = rectify_shape(mri_lesion_data)

        # sanitize lesion data to contain only single class
        def sanitize(lesion_data):
            lesion_data[lesion_data > 1] = 1
            if np.isnan(lesion_data).any():
                print('Lesion label of', ids[subject], 'contains NaN. Converting to 0.')
                lesion_data = np.nan_to_num(lesion_data)
            return lesion_data

        ct_lesion_outputs[subject, :, :, :] = sanitize(ct_lesion_data)
        if mri_lesion_paths:
            mri_lesion_outputs[subject, :, :, :] = sanitize(mri_lesion_data)

        brain_mask_image = nib.load(brain_mask_paths[subject])
        brain_mask_data = brain_mask_image.get_data()
        brain_mask_data = rectify_shape(brain_mask_data)

        if np.isnan(brain_mask_data).any():
            print('Brain mask of', ids[subject], 'contains NaN. Converting to 0.')
            brain_mask_data = np.nan_to_num(brain_mask_data)
        brain_masks[subject, :, :, :] = brain_mask_data

    if not mri_paths[0]:
        mri_inputs = []
        mri_lesion_outputs = []

    return ct_inputs, ct_lesion_outputs, mri_inputs, mri_lesion_outputs, brain_masks


def load_nifti(main_dir, ct_sequences, label_sequences, mri_sequences, mri_label_sequences, brain_mask_name, high_resolution = False):
    ids, ct_paths, ct_lesion_paths, mri_paths, mri_lesion_paths, brain_mask_paths = get_paths_and_ids(
        main_dir, ct_sequences, label_sequences,
        mri_sequences, mri_label_sequences,
        brain_mask_name)
    return (ids, load_images(ct_paths, ct_lesion_paths, mri_paths, mri_lesion_paths, brain_mask_paths, ids, high_resolution))

# Save data as compressed numpy array
def load_and_save_data(save_dir, main_dir, clinical_dir = None, clinical_name = None,
                       ct_sequences = [], label_sequences = [], mri_sequences = False,
                       external_memory=False, high_resolution = False):
    """
    Load data
        - Image data (from preprocessed Nifti)
        - Clinical data (from excel)

    Args:
        save_dir : directory to save data_to
        main_dir : directory containing images
        clinical_dir (optional) : directory containing clinical data (excel)
        ct_sequences (optional, array) : array with names of ct sequences
        label_sequences (optional, array) : array with names of VOI sequences
        mri_sequences (optional, boolean) : boolean determining if mri sequences should be included
        external_memory (optional, default False): on external memory usage, NaNs need to be converted to -1
        high_resolution (optional, default False): use non normalized images (patient space instead of MNI space)

    Returns:
        'clinical_data': numpy array containing the data for each of the patients [patient, (n_parameters)]
    """
    if len(ct_sequences) < 1:
        ct_sequences = ['wcoreg_Tmax', 'wcoreg_CBF', 'wcoreg_MTT', 'wcoreg_CBV']
        # ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_CBF', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV']
        # ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV']
        # ct_sequences = ['wcoreg_RAPID_TMax_[s]']
        if high_resolution:
            ct_sequences = ['coreg_Tmax', 'coreg_CBF', 'coreg_MTT', 'coreg_CBV']

    if len(label_sequences) < 1:
        # Import VOI GT with brain mask applied
        # to avoid False negatives in areas that cannot be predicted (as their are not part of the RAPID perf maps)
        label_sequences = ['masked_wcoreg_VOI']

        if high_resolution:
            label_sequences = ['masked_coreg_VOI']

    if mri_sequences:
        mri_sequences = ['wcoreg_t2_tse_tra', 'wcoreg_t2_TRACE', 'wcoreg_t2_ADC']
        # for MRI labeling, the mask should not be applied
        mri_label_sequences = ['wcoreg_VOI']

        if high_resolution:
            mri_sequences = ['coreg_t2_tse_tra', 'coreg_t2_TRACE', 'coreg_t2_ADC']
            # for MRI labeling, the mask should not be applied
            mri_label_sequences = ['coreg_VOI']
    else:
        mri_sequences = []
        mri_label_sequences = []

    brain_mask_name = 'brain_mask.nii'
    if high_resolution:
        brain_mask_name = 'hd_brain_mask.nii'

    print('Sequences used for CT', ct_sequences, label_sequences)
    print('Sequences used for MRI', mri_sequences, mri_label_sequences)

    included_subjects = np.array([])
    clinical_data = np.array([])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ids, (ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks) = load_nifti(main_dir, ct_sequences,
                                                                                        label_sequences, mri_sequences,
                                                                                        mri_label_sequences, brain_mask_name,
                                                                                        high_resolution)
    ids = np.array(ids)

    if clinical_dir is not None:
        included_subjects, clinical_data = load_clinical_data(ids, clinical_dir, clinical_name, external_memory=external_memory)

        # Remove patients with exclusion criteria
        ct_inputs = ct_inputs[included_subjects]
        ct_lesion_GT = ct_lesion_GT[included_subjects]
        mri_inputs = mri_inputs[included_subjects]
        mri_lesion_GT = mri_lesion_GT[included_subjects]

        print('Excluded', ids.shape[0] - ct_inputs.shape[0], 'subjects.')

    print('Saving a total of', ct_inputs.shape[0], 'subjects.')
    np.savez_compressed(os.path.join(save_dir, 'data_set'),
        params = {'ct_sequences': ct_sequences, 'ct_label_sequences': label_sequences,
                  'mri_sequences': mri_sequences, 'mri_label_sequences': mri_label_sequences},
        ids = ids, included_subjects = included_subjects,
        clinical_inputs = clinical_data, ct_inputs = ct_inputs, ct_lesion_GT = ct_lesion_GT,
        mri_inputs = mri_inputs, mri_lesion_GT = mri_lesion_GT, brain_masks = brain_masks)

def load_saved_data(data_dir, filename = 'data_set.npz'):
    params = np.load(os.path.join(data_dir, filename), allow_pickle=True)['params']
    ids = np.load(os.path.join(data_dir, filename), allow_pickle=True)['ids']
    clinical_inputs = np.load(os.path.join(data_dir, filename), allow_pickle=True)['clinical_inputs']
    ct_inputs = np.load(os.path.join(data_dir, filename), allow_pickle=True)['ct_inputs']
    ct_lesion_GT = np.load(os.path.join(data_dir, filename), allow_pickle=True)['ct_lesion_GT']
    mri_inputs = np.load(os.path.join(data_dir, filename), allow_pickle=True)['mri_inputs']
    mri_lesion_GT = np.load(os.path.join(data_dir, filename), allow_pickle=True)['mri_lesion_GT']
    brain_masks = np.load(os.path.join(data_dir, filename), allow_pickle=True)['brain_masks']

    print('Loading a total of', ct_inputs.shape[0], 'subjects.')
    print('Sequences used:', params)
    print(ids.shape[0] - ct_inputs.shape[0], 'subjects had been excluded.')

    return (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params)
