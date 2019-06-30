import os, sys
sys.path.insert(0, '../..')
import numpy as np
import nibabel as nib
import preprocessing.image_name_config as image_name_config
from analysis.visual import display

pct_sequences = image_name_config.pct_sequences

data_dir = '/Users/julian/errored'

def get_RAPID_4D_list(data_dir):
    '''
    get list of subjects with 4D RAPID maps
    :param data_dir: path to data
    :return: subjects_with_4D_maps
    '''
    subjects_with_4D_maps = []
    subject_folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in subject_folders:
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            studies = [o for o in os.listdir(modality_dir)]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if study.endswith('.nii') \
                        and ('Tmax' in study or 'CBF' in study or 'CBV' in study or 'MTT' in study)\
                        and not study.startswith('4D_'):
                    img = nib.load(study_path)
                    data = img.get_data()
                    if len(data.shape) != 3:
                        print(folder, study, len(data.shape))
                        subjects_with_4D_maps.append(folder)
                        break
    return subjects_with_4D_maps

def resolve_RAPID_4D_maps(data_dir):
    '''
    Verify that all RAPID maps are not 4D and try to fix it otherwise
    This can only be run if display works (X server)
    '''
    print('Watch out: this program overwrites the 4D RAPID files and replaces them with 3D versions (the old file is kept with a 4D_ prefix)')
    subjects_with_4D_maps = get_RAPID_4D_list(data_dir)
    if not subjects_with_4D_maps:
        print('All subjects have correct 3D maps.')
        return
    subject_folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in subject_folders:
        if not folder in subjects_with_4D_maps: continue
        prior_choice = None
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            studies = [o for o in os.listdir(modality_dir)]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if study.endswith('.nii') and ('Tmax' in study or 'CBF' in study or 'CBV' in study or 'MTT' in study):
                    img = nib.load(study_path)
                    data = img.get_data()
                    print(study, len(data.shape))
                    if prior_choice is None:
                        display(data[..., 0], block=False, title=folder + ' - dim 0')
                        display(data[..., 1], block=True, title=folder + ' - dim 1')
                        choice = input(folder + ': Choose first (0) or second dimension (1) [-1 for for skip]:\t')
                        prior_choice = choice
                    else:
                        # Be consistent with the previously chosen choice
                        choice = prior_choice
                    if choice == '-1': break
                    print('Reducing dimensions for ' + folder + 'to dimension: ', choice)
                    # Preserve the old file with a 4D_ prefix
                    os.rename(study_path, os.path.join(modality_dir, '4D_' + study))
                    # Keep only the selected dimension
                    reduced_data = np.squeeze(data[..., int(choice)])
                    coordinate_space = img.affine

                    reduced_img = nib.Nifti1Image(reduced_data, affine=coordinate_space)
                    nib.save(reduced_img, study_path)
    return


resolve_RAPID_4D_maps(data_dir)
