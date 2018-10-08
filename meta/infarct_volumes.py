import os
import pandas as pd
import nibabel as nib
import numpy as np

def extract_infarct_volumes(data_dir, save_data = True, save_dir = None):
    if save_dir is None:
            save_dir = data_dir

    mri_folder_prefix = 'Neuro_'
    lesion_GT_prefix = 'wcoreg_VOI_lesion'
    output_name = 'infarcted_volumes'

    df = pd.DataFrame(columns=('id', 'vox_volume', 'total_volume'))
    index = 0

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

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
                    if study.startswith(lesion_GT_prefix) and modality.startswith(mri_folder_prefix):
                        lesion_path = os.path.join(modality_dir, study)
                        lesion_image = nib.load(lesion_path)
                        lesion_data = lesion_image.get_data()
                        volume = np.sum(lesion_data)

                        df.loc[index] = [subject, volume, lesion_data.size]
                        index += 1

    if save_data:
        df.to_csv(os.path.join(save_dir, output_name + '.csv'))

    return df

def merge_with_clinical(clinical_path, volumes_path, merged_path = None):
    if merged_path is None:
        merged_path = os.path.join(os.getcwd(), 'merged_with_volumes.csv')
    clinical_df = pd.read_excel(clinical_path)
    volumes_df = pd.read_csv(volumes_path)
    merged_df = pd.merge(clinical_df, volumes_df, how='left', left_on='id_hospital_case', right_on='id')
    merged_df.to_csv(merged_path)
