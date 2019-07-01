import os
import numpy as np
import nibabel as nib


def binarize_lesion(masked_lesion_name, lesion_dir):
    '''
    Overwrite masked lesion with a binarized version
    :param masked_lesion_name:
    :param lesion_dir:
    :return:
    '''
    masked_lesion_path = os.path.join(lesion_dir, masked_lesion_name)
    lesion_img = nib.load(masked_lesion_path)
    lesion_data = lesion_img.get_data()
    coordinate_space = lesion_img.affine

    threshold = 0.8 * np.max(lesion_data)
    binary_data = np.zeros(lesion_data.shape)
    binary_data[lesion_data > threshold] = 1

    # keep continuous lesion with other name
    os.rename(os.path.join(lesion_dir, masked_lesion_name), os.path.join(lesion_dir, 'cont_' + masked_lesion_name))

    binary_img = nib.Nifti1Image(binary_data, affine=coordinate_space)
    nib.save(binary_img, os.path.join(masked_lesion_path))


def binarize_lesions_wrapper(data_dir):
    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]



        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isfile(os.path.join(modality_dir,o))]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if modality.startswith('MRI') & study.startswith('masked_wcoreg_VOI') & study.endswith('.nii'):
                    binarize_lesion(study, modality_dir)