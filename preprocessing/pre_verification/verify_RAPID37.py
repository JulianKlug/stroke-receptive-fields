import os, sys
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
import image_name_config as image_name_config

pct_sequences = image_name_config.pct_sequences
ct_perf_sequence_names = image_name_config.ct_perf_sequence_names

main_dir = '/Volumes/stroke_hdd1/stroke_db/2016/part1/'
data_dir = os.path.join(main_dir, '')

def verify_RAPID37(data_dir):
    '''
    Verify that patient with perfusion data have 37 RAPID perfCT images
    '''
    log_columns = ['subject', 'missing_RAPID_files_folder']
    missing_RAPIDs_df = pd.DataFrame(columns=log_columns)
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

            hasVPCT = 0
            for study in studies:
                if study in ct_perf_sequence_names: hasVPCT = 1

            if hasVPCT:
                TmaxComplete = 0
                MTTComplete = 0
                CBVComplete = 0
                CBFComplete = 0

                for study in studies:
                    study_dir = os.path.join(modality_dir, study)
                    if 'color' in study or not 'RAPID' in study:
                        continue
                    dcms = [f for f in os.listdir(os.path.join(modality_dir, study)) if f.endswith(".dcm") and not f.startswith('.')]
                    if len(dcms) != 37:
                        continue
                    if 'TMax' in study or 'Tmax' in study: TmaxComplete = 1
                    if 'MTT' in study: MTTComplete = 1
                    if 'CBV' in study: CBVComplete = 1
                    if 'CBF' in study: CBFComplete = 1
            if hasVPCT and (not TmaxComplete or not MTTComplete or not CBVComplete or not CBFComplete):
                print(folder, 'is missing complete 37 RAPID images for', modality)
                missing_RAPIDs_df = missing_RAPIDs_df.append(
                    pd.DataFrame([[folder, modality]],
                    columns = log_columns),
                    ignore_index=True)

    missing_RAPIDs_df.to_excel(os.path.join(data_dir, 'missing_RAPID_files.xlsx'))
    return missing_RAPIDs_df

verify_RAPID37(data_dir)
