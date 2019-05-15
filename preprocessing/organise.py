import os, subprocess, pydicom, datetime, re
from dateutil import parser
import numpy as np
import pandas as pd
import image_name_config
from unidecode import unidecode
import hashlib
from utils.naming_verification import tight_verify_name, loose_verify_name

main_dir = '/Volumes/stroke_hdd1/stroke_db/2016/'
data_dir = os.path.join(main_dir, 'additionnal')
output_dir = os.path.join(main_dir, 'extracted_additionnal')
enforce_VOI = True
copy = True
spc_ct_sequences = image_name_config.spc_ct_sequences
pct_sequences = image_name_config.pct_sequences
ct_perf_sequence_names = image_name_config.ct_perf_sequence_names
mri_sequences = image_name_config.mri_sequences
alternative_mri_sequences = image_name_config.alternative_mri_sequences
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
    subject_name_from_folder = '_'.join(unidecode(folder[:re.search("\d", folder).start() - 1].upper()).split(subject_name_seperator))

    modality_0 = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))][0]
    modality_0_path = os.path.join(dir, modality_0)
    studies = [o for o in os.listdir(modality_0_path)
                    if os.path.isdir(os.path.join(modality_0_path,o))]
    for study in studies:
        study_0_path = os.path.join(modality_0_path, study)
        dcms = [f for f in os.listdir(study_0_path) if f.endswith(".dcm")]
        if not dcms: continue
        dcm = pydicom.dcmread(os.path.join(study_0_path, dcms[0]))

        full_name = '_'.join(re.split(r'[/^ ]', unidecode(str(dcm.PatientName).upper())))
        last_name = unidecode(str(dcm.PatientName).split('^')[0].upper())
        first_name = unidecode(str(dcm.PatientName).split('^')[1].upper())

        def flatten_string(string):
            return unidecode(''.join(re.split(r'[,-]', str(string)))).upper()

        attached_full_name = '_'.join(flatten_string(last_name).split(' ')) + '_' + flatten_string(first_name)
        patient_birth_date = dcm.PatientBirthDate

        if subject_name_from_folder != full_name and subject_name_from_folder != attached_full_name :
            print(first_name, last_name, full_name, attached_full_name)
            raise Exception('Names do not match between folder name and name in dicom',
                subject_name_from_folder, full_name)

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

            hasSPC = 0
            hasPCT_maps = 0

            for study in studies:
                study_dir = os.path.join(modality_dir, study)

                if loose_verify_name(study, pct_sequences):
                    dcms = [f for f in os.listdir(study_dir) if f.endswith(".dcm") and not f.startswith('.')]
                    if not 'color' in study and 'RAPID' in study and len(dcms) >= 37:
                        hasPCT_maps = 1

                if loose_verify_name(study, spc_ct_sequences): hasSPC = 1

                if hasSPC and hasPCT_maps:
                    # verify CT is unique
                    if pCT_found == 1:
                        message = 'Two perfusion CTs found'
                        print(folder, message)
                        error_log_df = error_log_df.append(
                            pd.DataFrame([[folder, True, False, message]], columns = error_log_columns),
                            ignore_index=True)
                        break

                    dcms = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
                    dcm = pydicom.dcmread(os.path.join(study_dir, dcms[0]))

                    pct_date = datetime.datetime.combine(parser.parse(dcm.StudyDate), parser.parse(dcm.StudyTime).time())

                    imaging_info.update({subject_key : {
                        'pct_date' : pct_date,
                        'pct_path' : modality_dir
                        }})
                    pCT_found = 1
                    break

    return imaging_info, error_log_df

def choose_correct_MRI(dir, pct_date):
    """
    go trough subject directory and choose the MRI corresponding to the labeled lesion
    it should be the earliest after the pct was done that has T2 and DWI sequences
    returns : MRI_found, MRI_path, MRI_date, multiple_mri_studies_found, VOI_found
    """
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
        if len(dcms) == 0:
            continue
        dcm = pydicom.dcmread(os.path.join(os.path.join(modality_dir, studies[0]), dcms[0]))
        modality_date = datetime.datetime.combine(parser.parse(dcm.StudyDate), parser.parse(dcm.StudyTime).time())
        # don't take into account if MRI was done before the CT
        if modality_date < pct_date:
            continue
        for study in studies:
            study_dir = os.path.join(modality_dir, study)
            if tight_verify_name(study, mri_sequences):
                hasT2 = 1
            if 'ADC' in study or 'TRACE' in study or 'adc' in study \
                or 'trace' in study or 'DWI' in study or 'dwi' in study or 'b1000' in study:
                hasDWI = 1
            if 'VOI' in study or 'lesion' in study or 'Lesion' in study:
                lesionDrawn = 1
                return (True, modality_dir, modality_date, False, True)
        if hasT2 and hasDWI:
            mri_dates.append(modality_date)
            mri_paths.append(modality_dir)
    if not mri_dates:
        return (False, [], [], False, False)
    if len(mri_dates) > 1:
        multiple_mri_studies_found = True
    earliest_complete_mri_after_pct = np.argmin(mri_dates)
    return (True, mri_paths[earliest_complete_mri_after_pct], mri_dates[earliest_complete_mri_after_pct], multiple_mri_studies_found, False)

def add_MRI_paths_and_date(dir, imaging_info, error_log_df):
    # add MRI info (date and path) to imaging_info
    folders = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]
    for folder in folders:
        folder_dir = os.path.join(dir, folder)
        (last_name, first_name, patient_birth_date) = get_subject_info(folder_dir)
        subject_key = last_name + '^' + first_name + '^' + patient_birth_date
        # Skip patients with no perfusion CT
        if not subject_key in imaging_info:
            continue
        pct_date = imaging_info[subject_key]['pct_date']

        MRI_found, MRI_path, MRI_date, multiple_mri_studies_found, VOI_found = choose_correct_MRI(folder_dir, pct_date)
        if not MRI_found:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, False, True, 'MRI not found']], columns = error_log_columns),
                ignore_index=True)
            # if no MRI for this subject, remove it from the list of usable subjects
            del imaging_info[subject_key]
            continue
        if multiple_mri_studies_found:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, False, False, 'multiple MRI with T2 and DWI found']], columns = error_log_columns),
                ignore_index=True)
        imaging_info[subject_key].update({
            'mri_path': MRI_path,
            'mri_date': MRI_date,
            'VOI_found': VOI_found
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
            if copy: subprocess.run(['cp', '-rf', file_path, new_file_path])
            move_log_df = move_log_df.append(
                pd.DataFrame([[patient_identifier, file_path, new_file_path]], columns = move_log_columns),
                ignore_index=True)

    # select CT files
    ct_studies = [o for o in os.listdir(ct_folder_path)
                    if os.path.isdir(os.path.join(ct_folder_path,o))]
    selected_ct_study_paths = []
    for ct_study in ct_studies:
        ct_study_path = os.path.join(ct_folder_path, ct_study)

        # find SPC
        if loose_verify_name(ct_study, spc_ct_sequences):
            selected_ct_study_paths.append(ct_study_path)

        if 'color' in ct_study or not 'RAPID' in ct_study:
            continue
        dcms = [f for f in os.listdir(os.path.join(ct_folder_path, ct_study)) if f.endswith(".dcm") and not f.startswith('.')]
        # exclude pCTs with something else than 37 images
        if len(dcms) < 37:
            continue

        if loose_verify_name(ct_study, pct_sequences):
            selected_ct_study_paths.append(ct_study_path)

    # select MRI files
    # find T2w sequence that VOI was drawn on
    mri_studies = [o for o in os.listdir(mri_folder_path)
                    if os.path.isdir(os.path.join(mri_folder_path,o)) or o.endswith('.nii')]
    selected_mri_study_paths = []
    for mri_study in mri_studies:
        mri_study_path = os.path.join(mri_folder_path, mri_study)
        if tight_verify_name(mri_study, mri_sequences):
            selected_mri_study_paths.append(mri_study_path)

        if 'VOI' in mri_study or 'lesion' in mri_study or 'Lesion' in mri_study:
            new_file_name = 'VOI_' + patient_identifier + '.nii'
            new_file_path = os.path.join(patient_output_folder, new_file_name)
            if not os.path.exists(new_file_path):
                if copy: subprocess.run(['cp', '-rf', mri_study_path, new_file_path])
                move_log_df = move_log_df.append(
                    pd.DataFrame([[patient_identifier, mri_study_path, new_file_path]], columns = move_log_columns),
                    ignore_index=True)

    # if no MRI with primary sequence found, try with secondary sequence
    if not selected_mri_study_paths: # ie not mri study found yet
        for mri_study in mri_studies:
            mri_study_path = os.path.join(mri_folder_path, mri_study)
            if tight_verify_name(mri_study, alternative_mri_sequences):
                selected_mri_study_paths.append(mri_study_path)

    print(selected_mri_study_paths)

    selected_study_paths = selected_mri_study_paths + selected_ct_study_paths
    for selected_study_path in selected_study_paths:
        selected_study_name = os.path.basename(selected_study_path)
        # rename to constant name space
        if selected_study_path in selected_mri_study_paths:
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

        if loose_verify_name(selected_study_name, spc_ct_sequences):
            new_study_name = 'SPC_301mm_Std' + '_' + patient_identifier

        output_modality_dir = os.path.join(patient_output_folder, modality_name)
        if not os.path.exists(output_modality_dir):
            os.makedirs(output_modality_dir)

        new_study_path = os.path.join(output_modality_dir, new_study_name)
        if not os.path.exists(new_study_path):
            if copy: subprocess.run(['cp', '-rf', selected_study_path, new_study_path])
            move_log_df = move_log_df.append(
                pd.DataFrame([[patient_identifier, selected_study_path, new_study_path]], columns = move_log_columns),
                ignore_index=True)

    return move_log_df

def enforce_VOI_presence(dir, imaging_info, error_log_df):
    # exclude patients with no VOI
    folders = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]
    for folder in folders:
        folder_dir = os.path.join(dir, folder)
        (last_name, first_name, patient_birth_date) = get_subject_info(folder_dir)
        subject_key = last_name + '^' + first_name + '^' + patient_birth_date
        # Skip patients with no perfusion CT or MRI
        if not subject_key in imaging_info:
            continue
        # Skip patients where VOI has already been found
        if imaging_info[subject_key]['VOI_found']:
            continue

        VOI_candidates_subj_dir = [f for f in os.listdir(folder_dir)
                    if f.endswith(".nii") and ('VOI' in f or 'lesion' in f or 'Lesion' in f)]
        mri_dir = imaging_info[subject_key]['mri_path']
        VOI_candidates_mri_dir = [f for f in os.listdir(mri_dir)
                    if f.endswith(".nii") and ('VOI' in f or 'lesion' in f or 'Lesion' in f)]
        if not VOI_candidates_subj_dir and not VOI_candidates_mri_dir:
            error_log_df = error_log_df.append(
                pd.DataFrame([[folder, False, True, 'no VOI found']], columns = error_log_columns),
                ignore_index=True)
            del imaging_info[subject_key]
    return imaging_info, error_log_df

def main(dir, output_dir):
    # initialise logs
    error_log_df = pd.DataFrame(columns=error_log_columns)
    move_log_df = pd.DataFrame(columns=move_log_columns)
    anonymisation_columns = ['patient_identifier', 'anonymised_id', 'original_ct_path', 'ct_date', 'original_mri_path', 'mri_date']
    anonymisation_df = pd.DataFrame(columns=anonymisation_columns)

    # get CT and MRI paths and image info
    print('Extracting CT paths and dates')
    imaging_info, error_log_df = get_ct_paths_and_date(dir, error_log_df)
    print(len(imaging_info), 'subjects selected based on CT')
    print('Extracting MRI paths and dates')
    imaging_info, error_log_df = add_MRI_paths_and_date(dir, imaging_info, error_log_df)
    print(len(imaging_info), 'subjects selected based on MRI and CT')

    if enforce_VOI:
        imaging_info, error_log_df = enforce_VOI_presence(dir, imaging_info, error_log_df)

    for patient_identifier in imaging_info:
        # hash id for anonymisation
        ID = hashlib.sha256(patient_identifier.encode('utf-8')).hexdigest()[:8]
        pid = 'subj-' + str(ID)
        print('Copying data for', patient_identifier)
        patient_output_folder = os.path.join(output_dir, pid)
        if os.path.exists(patient_output_folder):
            print(patient_identifier, ': Namespace already taken by', pid)
            print('Data not copied')
            error_log_df = error_log_df.append(
                pd.DataFrame([[patient_identifier, True, True, 'PID already taken']], columns = error_log_columns),
                ignore_index=True)
            continue
        move_log_df = move_selected_patient_data(pid, imaging_info[patient_identifier]['pct_path'], imaging_info[patient_identifier]['mri_path'], output_dir, move_log_df)
        anonymisation_df = anonymisation_df.append(
            pd.DataFrame([[patient_identifier, pid,
                imaging_info[patient_identifier]['pct_path'], imaging_info[patient_identifier]['pct_date'], imaging_info[patient_identifier]['mri_path'], imaging_info[patient_identifier]['mri_date']
                ]], columns = anonymisation_columns),
            ignore_index=True)

    error_log_df.to_excel(os.path.join(dir, 'reorganisation_error_log.xlsx'))
    move_log_df.to_excel(os.path.join(dir, 'reorganisation_path_log.xlsx'))
    anonymisation_df.to_excel(os.path.join(dir, 'anonymisation_key.xlsx'))

    # todo check integrity of patients

main(data_dir, output_dir)
