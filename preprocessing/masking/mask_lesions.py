import os
import subprocess
import numpy as np
import nibabel as nib


def mask_lesion(lesion_name, lesion_dir, lesion_mask_path, brain_mask_path):
    lesion_img = nib.load(lesion_mask_path)
    lesion_data = lesion_img.get_data()
    coordinate_space = lesion_img.affine
    image_extension = '.nii'

    brain_mask_image = nib.load(brain_mask_path)
    brain_mask_data = brain_mask_image.get_data()

    masked_data = brain_mask_data * lesion_data

    masked_img = nib.Nifti1Image(masked_data, affine=coordinate_space)
    nib.save(masked_img, os.path.join(lesion_dir,  'masked_' + lesion_name))

# Apply a brain mask to the MRI lesion maps to avoid having parts of the lesion outside of brain available in RAPID maps
def mask_lesions_wrapper(data_dir):
    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        lesion_mask_path = [];
        brain_mask_path = [];
        lesion_mask = [];

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isfile(os.path.join(modality_dir,o))]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if modality.startswith('MRI') & study.startswith('wcoreg_VOI'):
                    lesion_mask_path.append(study_path)
                    lesion_mask.append(study)
                    lesion_dir = modality_dir
                if modality.startswith('pCT') & study.startswith('brain_mask'):
                    brain_mask_path.append(study_path)

        if (not (lesion_mask_path) or not (brain_mask_path)):
            print('Not all images found for subject ', subject)
        else:
            print('Processing subject ', subject)

            mask_lesion(lesion_mask[0], lesion_dir, lesion_mask_path[0], brain_mask_path[0])

            # try:
            # except :
            #     print('Not all images found for subject ', subject)



            # try:
            #     print('Processing subject ', subject)
            #     subprocess.run(['fslmaths', lesion_mask_path, '-mul', brain_mask_path, 'masked_' + lesion_mask], cwd = modality_dir)
            #     subprocess.run(['gunzip', os.path.join('Neuro_Cerebrale_64Ch', 'masked_' + lesion_mask)], cwd = modality_dir)
            #     subprocess.run(['rm', os.path.join('Neuro_Cerebrale_64Ch', 'masked_' + lesion_mask + '.gz')], cwd = modality_dir)
            #
            # except :
            #     print('Not all images found for subject ', subject)
