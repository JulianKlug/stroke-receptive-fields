import os
import numpy as np
import nibabel as nib


def binarize_mask(mask_file_name, mask_dir, threshold=0.5, relative_to_max=False):
    '''
    Overwrite mask with a binarized version
    :param masked_lesion_name:
    :param lesion_dir:
    :param threshold: threshold to apply to
    :param relative_to_max: if true use threshold as a fraction of maximum, else use as absolute value
    :return:
    '''
    print(mask_file_name)
    masked_lesion_path = os.path.join(mask_dir, mask_file_name)
    lesion_img = nib.load(masked_lesion_path)
    lesion_data = lesion_img.get_data()
    coordinate_space = lesion_img.affine

    if np.array_equal(lesion_data, lesion_data.astype(bool)):
        print('Already binary. Skipping.')
        return

    if np.isnan(lesion_data).any():
        print('CT images of', mask_file_name, 'contains NaN. Converting to 0.')
        lesion_data = np.nan_to_num(lesion_data)

    if relative_to_max:
        threshold = threshold * np.max(lesion_data)

    binary_data = np.zeros(lesion_data.shape)
    binary_data[lesion_data > threshold] = 1

    # keep continuous lesion with other name
    os.rename(os.path.join(mask_dir, mask_file_name), os.path.join(mask_dir, 'cont_' + mask_file_name))

    binary_img = nib.Nifti1Image(binary_data, affine=coordinate_space)
    nib.save(binary_img, os.path.join(masked_lesion_path))


def binarize_masks_wrapper(data_dir, masked_VOI=True, high_resolution=False):
    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    VOI_start = 'wcoreg_VOI'
    vessel_mask_start = 'wmask_filtered_extracted_betted_Angio'

    if high_resolution:
        VOI_start = 'coreg_VOI'
        vessel_mask_start = 'mask_filtered_extracted_betted_Angio'

    # ie mask lesions has been called before (necessary if used with RAPID perf maps)
    if masked_VOI:
        VOI_start = 'masked_' + VOI_start

    for subject in subjects:
        print('Processing', subject)
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]



        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isfile(os.path.join(modality_dir,o))]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if modality.startswith('MRI') & study.startswith(VOI_start) & study.endswith('.nii'):
                    binarize_mask(study, modality_dir, threshold=0.8, relative_to_max=True)
                if modality.startswith('pCT') & study.startswith(vessel_mask_start) & study.endswith('.nii'):
                    binarize_mask(study, modality_dir, threshold=0.2, relative_to_max=False)

