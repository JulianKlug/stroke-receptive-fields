import os
import nibabel as nib
import numpy as np


# provided a given directory return list of paths to ct_sequences and lesion_maps
def get_paths(data_dir, ct_sequences, mri_sequences):

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    lesion_paths = []
    ct_paths = []

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        ct_channels = []
        lesion_map = []

        if os.path.isdir(subject_dir):
            modalities = [o for o in os.listdir(subject_dir)
                            if os.path.isdir(os.path.join(subject_dir,o))]

            for modality in modalities:
                modality_dir = os.path.join(subject_dir, modality)

                studies = os.listdir(modality_dir)

                for study in studies:
                    if study.startswith(tuple(mri_sequences)):
                        lesion_map.append(os.path.join(modality_dir, study))

                    if study.startswith(tuple(ct_sequences)):
                        ct_channels.append(os.path.join(modality_dir, study))

        if len(ct_sequences) == len(ct_channels) and len(mri_sequences) == len(lesion_map):
            lesion_paths.append(lesion_map[0])
            ct_paths.append(ct_channels)
            print('Adding', subject)
        else :
            print('Not all images found for this subject. Skipping.', subject)

    return (ct_paths, lesion_paths)

# Load nifi image maps from paths (first image is used as reference for dimensions)
# - ct_paths : list of lists of paths of channels
# - lesion_paths : list of paths of lesions maps
# - return two lists containing image data for cts (as 4D array) and lesion_maps
def load_images(ct_paths, lesion_paths):
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

    for subject in range(len(ct_paths)):
        ct_channels = ct_paths[subject]
        for c in range(n_c):
            image = nib.load(ct_channels[c])
            image_data = image.get_data()
            if first_image_data.shape != image_data.shape:
                raise ValueError('Image does not have correct dimensions.', ct_channels[c])

            ct_inputs[subject, :, :, :, c] = image_data

        lesion_image = nib.load(lesion_paths[subject])
        lesion_data = lesion_image.get_data()

        lesion_outputs[subject, :, :, :] = lesion_data

    return (ct_inputs, lesion_outputs)


def load_nifti(main_dir, ct_sequences, mri_sequences):
    ct_paths, lesion_paths = get_paths(main_dir, ct_sequences, mri_sequences)
    return load_images(ct_paths, lesion_paths)


# Save data as compressed numpy array
def load_and_save_data(data_dir, main_dir, ct_sequences = [], mri_sequences = []):
    if len(ct_sequences) < 1:
        ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
        # ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
        # ct_sequences = ['wcoreg_RAPID_TMax_[s]']
    if len(mri_sequences) < 1:
        mri_sequences = ['wcoreg_VOI_lesion']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ct_inputs, lesion_GT = load_nifti(main_dir, ct_sequences, mri_sequences)
    np.savez_compressed(os.path.join(data_dir, 'data_set'), ct_inputs = ct_inputs, lesion_GT = lesion_GT)

def load_saved_data(data_dir):
    ct_inputs = np.load(os.path.join(data_dir, 'data_set.npz'))['ct_inputs']
    lesion_GT = np.load(os.path.join(data_dir, 'data_set.npz'))['lesion_GT']
    return (ct_inputs, lesion_GT)
