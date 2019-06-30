import os
import numpy as np
import pandas as pd

main_dir = '/Volumes/stroke_hdd1/stroke_db/2017/imaging_data'
data_dir = os.path.join(main_dir, '')

def find_empty_folders(data_dir):
    log_columns = ['subject', 'empty_folder']
    empty_folders_df = pd.DataFrame(columns=log_columns)
    subject_folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in subject_folders:
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]
            if [f for f in os.listdir(modality_dir) if not f.startswith('.')] == []:
                empty_folders_df = empty_folders_df.append(
                    pd.DataFrame([[folder, os.path.join(modality)]],
                    columns = log_columns),
                    ignore_index=True)

            for study in studies:
                study_dir = os.path.join(modality_dir, study)
                if [f for f in os.listdir(study_dir) if not f.startswith('.')] == []:
                    empty_folders_df = empty_folders_df.append(
                        pd.DataFrame([[folder, os.path.join(modality, study)]],
                        columns = log_columns),
                        ignore_index=True)

    empty_folders_df.to_excel(os.path.join(data_dir, 'empty_folders.xlsx'))

find_empty_folders(data_dir)
