import os
import subprocess
import pydicom
import datetime
from dateutil import parser
import numpy as np
import pandas as pd

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'working_data')
output_dir = os.path.join(main_dir, 'reorganised_test2')
spc_ct_sequences = ['SPC_301mm_Std', 'SPC 3.0-1mm Std']
pct_sequences = ['TMax', 'Tmax', 'MTT', 'CBV', 'CBF']
ct_perf_sequence_names = ['VPCT_Perfusion_4D_50_Hr36', 'VPCT Perfusion 4D 5.0 Hr36']
mri_sequences = ['t2_tse_tra', 'T2W_TSE_tra']
subject_name_seperator = '_'
error_log_columns = ['folder', 'error', 'exclusion', 'message']
move_log_columns = ['folder', 'initial_path', 'new_path']

# Logic
# Check if patient has a pCT
# Check date of pCT
# Take MRI that comes just after pCT and contains right sequences

def get_subject_info(dir):
    # from a given subject folder
    # extract last_name, first_name, patient_birth_date

    # verify that name on folder corresponds
    folder = os.path.basename(dir)
    folder_subject_first_name = folder.split(subject_name_seperator)[1].upper()
    folder_subject_last_name = folder.split(subject_name_seperator)[0].upper()

    modality_0 = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))][0]
    modality_0_path = os.path.join(dir, modality_0)
    study_0 = [o for o in os.listdir(modality_0_path)
                    if os.path.isdir(os.path.join(modality_0_path,o))][0]
    study_0_path = os.path.join(modality_0_path, study_0)
    dcms = [f for f in os.listdir(study_0_path) if f.endswith(".dcm")]
    dcm = pydicom.dcmread(os.path.join(study_0_path, dcms[0]))

    last_name = str(dcm.PatientName).split('^')[0].upper()
    first_name = str(dcm.PatientName).split('^')[1].upper()
    patient_birth_date = dcm.PatientBirthDate

    if folder_subject_first_name != first_name or folder_subject_last_name != last_name:
        raise Exception('Names do not match between folder name and name in dicom',
            first_name, last_name, folder_subject_first_name, folder_subject_last_name)

    return (last_name, first_name, patient_birth_date)

def get_ct_paths_and_date(dir, error_log_df):
    imaging_info = {}
    folders = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]

    for folder in folders:
        folder_dir = os.path.join(data_dir, folder)

        try:
            (last_name, first_name, patient_birth_date) = get_subject_info(folder_dir)
            subject_key = last_name + '^' + first_name + '^' + patient_birth_date
            if subject_key in imaging_info:
                raise Exception('Patient names collision', folder, subject_key)
        except Exception as e:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, True, True, e]], columns = error_log_columns),
                ignore_index=True)
            continue

        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]

        pCT_found = 0
        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)

            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]

            for study in studies:
                study_dir = os.path.join(modality_dir, study)

                if study in ct_perf_sequence_names:
                    # verify CT is unique
                    if pCT_found == 1:
                        message = 'Two perfusion CTs found'
                        error_log_df = error_log_df.append(
                            pd.DataFrame([[folder, True, False, message]], columns = error_log_columns),
                            ignore_index=True)
                        break
                        # TODO: handle exception

                    dcms = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
                    dcm = pydicom.dcmread(os.path.join(study_dir, dcms[0]))

                    pct_date = datetime.datetime.combine(parser.parse(dcm.StudyDate), parser.parse(dcm.StudyTime).time())

                    imaging_info.update({subject_key : {
                        'pct_date' : pct_date,
                        'pct_path' : modality_dir
                        }})
                    pCT_found = 1
                    break
            # TODO: extract pCT paths
    return imaging_info, error_log_df

def choose_correct_MRI(dir, pct_date):
    # go trough subject directory and choose the MRI corresponding to the labeled lesion
    # it should be the earliest after the pct was done that has T2 and DWI sequences
    modalities = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]
    mri_dates = []
    mri_paths = []
    multiple_mri_studies_found = False
    for modality in modalities:
        hasT2 = 0
        hasDWI = 0
        modality_dir = os.path.join(dir, modality)

        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isdir(os.path.join(modality_dir,o))]
        dcms = [f for f in os.listdir(os.path.join(modality_dir, studies[0])) if f.endswith(".dcm")]
        dcm = pydicom.dcmread(os.path.join(os.path.join(modality_dir, studies[0]), dcms[0]))
        modality_date = datetime.datetime.combine(parser.parse(dcm.StudyDate), parser.parse(dcm.StudyTime).time())
        # don't take into account if MRI was done before the CT
        if modality_date < pct_date:
            continue

        for study in studies:
            study_dir = os.path.join(modality_dir, study)
            if 'T2' in study or 't2' in study:
                hasT2 = 1
            if 'ADC' in study or 'TRACE' in study or 'adc' in study \
                or 'trace' in study or 'DWI' in study or 'dwi' in study:
                hasDWI = 1
            if 'VOI' in study or 'lesion' in study or 'Lesion' in study:
                lesionDrawn = 1
                mri_path = modality_dir
                return mri_path
        if hasT2 and hasDWI:
            mri_dates.append(modality_date)
            mri_paths.append(modality_dir)
    if not mri_dates:
        return (False, [], [], False)
    if len(mri_dates) > 1:
        multiple_mri_studies_found = True
    earliest_complete_mri_after_pct = np.argmin(mri_dates)
    return (True, mri_paths[earliest_complete_mri_after_pct], mri_dates[earliest_complete_mri_after_pct], multiple_mri_studies_found)

def add_MRI_paths_and_date(dir, imaging_info, error_log_df):
    # add MRI info (date and path) to imaging_info
    folders = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]
    for folder in folders:
        folder_dir = os.path.join(data_dir, folder)

        (last_name, first_name, patient_birth_date) = get_subject_info(folder_dir)
        subject_key = last_name + '^' + first_name + '^' + patient_birth_date
        # Skip patients with no perfusion CT
        if not subject_key in imaging_info:
            continue
        pct_date = imaging_info[subject_key]['pct_date']

        MRI_found, MRI_path, MRI_date, multiple_mri_studies_found = choose_correct_MRI(folder_dir, pct_date)
        if not MRI_found:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, False, True, 'MRI not found']], columns = error_log_columns),
                ignore_index=True)
            continue
        if multiple_mri_studies_found:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, False, False, 'multiple MRI with T2 and DWI found']], columns = error_log_columns),
                ignore_index=True)
        imaging_info[subject_key].update({
            'mri_path': MRI_path,
            'mri_date': MRI_date
        })
    return (imaging_info, error_log_df)

def move_selected_patient_data(patient_identifier, ct_folder_path, mri_folder_path, output_dir, move_log_df):
    patient_output_folder = os.path.join(output_dir, patient_identifier)
    if not os.path.exists(patient_output_folder):
        os.makedirs(patient_output_folder)

    # find VOI if in main patient dir
    patient_folder = os.path.dirname(ct_folder_path)
    VOI_candidates = [f for f in os.listdir(patient_folder)
                if f.endswith(".nii") and ('VOI' in f or 'lesion' in f or 'Lesion' in f)]
    if VOI_candidates:
        file_path = os.path.join(patient_folder, VOI_candidates[0])
        new_file_name = 'VOI_' + patient_identifier + '.nii'
        new_file_path = os.path.join(patient_output_folder, new_file_name)
        if not os.path.exists(new_file_path):
            subprocess.run(['cp', '-rf', file_path, new_file_path])
            move_log_df = move_log_df.append(
                pd.DataFrame([[patient_identifier, file_path, new_file_path]], columns = move_log_columns),
                ignore_index=True)

    # select CT files
    ct_studies = [o for o in os.listdir(ct_folder_path)
                    if os.path.isdir(os.path.join(ct_folder_path,o))]
    selected_ct_study_paths = []
    for ct_study in ct_studies:
        ct_study_path = os.path.join(ct_folder_path, ct_study)
        if ct_study in spc_ct_sequences:
            selected_ct_study_paths.append(ct_study_path)

        if 'color' in ct_study or not 'RAPID' in ct_study:
            continue
        dcms = [f for f in os.listdir(os.path.join(ct_folder_path, ct_study)) if f.endswith(".dcm")]
        # exclude pCTs with something else than 37 images
        if len(dcms) != 37:
            continue
        for pct_seq in pct_sequences:
            if pct_seq in ct_study:
                selected_ct_study_paths.append(ct_study_path)

    # select MRI files
    mri_studies = [o for o in os.listdir(mri_folder_path)
                    if os.path.isdir(os.path.join(mri_folder_path,o))]
    selected_mri_study_paths = []
    for mri_study in mri_studies:
        mri_study_path = os.path.join(mri_folder_path, mri_study)
        if mri_study in mri_sequences:
            selected_ct_study_paths.append(mri_study_path)
        if 'VOI' in mri_study or 'lesion' in mri_study or 'Lesion' in mri_study:
            new_file_name = 'VOI_' + patient_identifier + '.nii'
            new_file_path = os.path.join(patient_output_folder, new_file_name)
            if not os.path.exists(new_file_path):
                subprocess.run(['cp', '-rf', mri_study_path, new_file_path])
                move_log_df = move_log_df.append(
                    pd.DataFrame([[patient_identifier, mri_study_path, new_file_path]], columns = move_log_columns),
                    ignore_index=True)

    selected_study_paths = selected_mri_study_paths + selected_ct_study_paths
    for selected_study_path in selected_study_paths:
        selected_study_name = os.path.basename(selected_study_path)
        # rename to constant name space
        if selected_study_name in mri_sequences:
            new_study_name = 't2_tse_tra' + '_' + patient_identifier
            modality_name = 'MRI'
        else:
            modality_name = 'pCT'

        if 'Tmax' in selected_study_name or 'TMax' in selected_study_name:
            new_study_name = 'Tmax' + '_' + patient_identifier
        if 'MTT' in selected_study_name:
            new_study_name = 'MTT' + '_' + patient_identifier
        if 'CBF' in selected_study_name:
            new_study_name = 'CBF' + '_' + patient_identifier
        if 'CBV' in selected_study_name:
            new_study_name = 'CBV' + '_' + patient_identifier

        if selected_study_name in spc_ct_sequences:
            new_study_name = 'SPC_301mm_Std' + '_' + patient_identifier

        output_modality_dir = os.path.join(patient_output_folder, modality_name)
        if not os.path.exists(output_modality_dir):
            os.makedirs(output_modality_dir)

        new_study_path = os.path.join(output_modality_dir, new_study_name)
        if not os.path.exists(new_study_path):
            subprocess.run(['cp', '-rf', selected_study_path, new_study_path])
            move_log_df = move_log_df.append(
                pd.DataFrame([[patient_identifier, selected_study_path, new_study_path]], columns = move_log_columns),
                ignore_index=True)

    return move_log_df


def main(dir, output_dir):
    error_log_df = pd.DataFrame(columns=error_log_columns)
    move_log_df = pd.DataFrame(columns=move_log_columns)
    anonymisation_columns = ['patient_identifier', 'anonymised_id', 'original_ct_path', 'ct_date', 'original_mri_path', 'mri_date']
    anonymisation_df = pd.DataFrame(columns=anonymisation_columns)
    imaging_info, error_log_df = get_ct_paths_and_date(dir, error_log_df)
    imaging_info, error_log_df = add_MRI_paths_and_date(dir, imaging_info, error_log_df)
    id = 0
    for patient_identifier in imaging_info:
        # use sequential id for anonymisation
        pid = 'subj' + str(id)
        move_log_df = move_selected_patient_data(pid, imaging_info[patient_identifier]['pct_path'], imaging_info[patient_identifier]['mri_path'], output_dir, move_log_df)
        anonymisation_df = anonymisation_df.append(
            pd.DataFrame([[patient_identifier, pid,
                imaging_info[patient_identifier]['pct_path'], imaging_info[patient_identifier]['pct_date'], imaging_info[patient_identifier]['mri_path'], imaging_info[patient_identifier]['mri_date']
                ]], columns = anonymisation_columns),
            ignore_index=True)
        id += 1

    error_log_df.to_excel(os.path.join(dir, 'reorganisation_error_log.xlsx'))
    move_log_df.to_excel(os.path.join(dir, 'reorganisation_path_log.xlsx'))
    anonymisation_df.to_excel(os.path.join(dir, 'anonymisation_key.xlsx'))

    # todo check integrity of patients

main(data_dir, output_dir)
