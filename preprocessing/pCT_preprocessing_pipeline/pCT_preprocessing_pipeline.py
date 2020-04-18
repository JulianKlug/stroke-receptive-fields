import os, time, argparse, sys, shutil
import pandas as pd
sys.path.insert(0, '../')
from tools.fsl_wrappers.mcflirt import mcflirt
from tools.coregistration_4D import coregistration_4D
from tools.segmentation.brain_extraction import brain_extraction


def pCT_preprocessing_pipeline(data_dir, reverse_reading, CT_dirname='pCT',
                                pCT_name='VPCT', spc_name='SPC_301mm',
                                brain_mask_name='betted_SPC_301mm', brain_mask_suffix='_Mask.nii.gz'):
    '''
    Preprocessing pipeline for 4D perfusion CT
        - 1. Motion correction
        - 2. Realignement to non contrast anatomical (spc)
        - 3. Brain extraction
    Output: final preprocessed files get a p_ prefix, intermediary files are destroyed
    Logs: pCT_preprocessing_error_log_TIMESTAMP.xslsx
    :param data_dir: directory containing data
    :param reverse_reading: read directory in reverse order
    :param CT_dirname: name of subdir in subject containing CT data
    :param pCT_name: prefix of CT file
    :param spc_name: prefix on anatomical non contrast CT
    :param brain_mask_name: prefix of brain mask image
    :param brain_mask_suffix: suffix of brain mask image
    :return:
    '''
    error_log_columns = ['subject', 'message', 'excluded']
    error_log_df = pd.DataFrame(columns=error_log_columns)
    timestamp = str(time.time()).split('.')[0]

    subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, o))]

    # allow for simultaneous processing
    if reverse_reading:
        subjects.reverse()

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            if not modality.startswith(CT_dirname):
                continue
            spc_files = [i for i in os.listdir(modality_dir) if
                         os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(spc_name)
                         and i.endswith('.nii')]
            pCT_files = [i for i in os.listdir(modality_dir) if
                         os.path.isfile(os.path.join(modality_dir, i))
                         and i.startswith(pCT_name + '_' + subject + '.nii')
                         and i.endswith('.nii')]
            brain_mask_files = [i for i in os.listdir(modality_dir) if
                                os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(brain_mask_name)
                                and i.endswith(brain_mask_suffix)]

            if len(spc_files) < 1:
                print('No SPC file found', subject, spc_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'no SPC', True]], columns=error_log_columns), ignore_index=True)
                continue
            if len(spc_files) > 1:
                print('Multiple SPC files found', subject, spc_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'multiple SPC', False]], columns=error_log_columns), ignore_index=True)
            if len(pCT_files) < 1:
                print('No pCT file found', subject, pCT_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'no pCT', True]], columns=error_log_columns), ignore_index=True)
                continue
            if len(pCT_files) > 1:
                print('Multiple pCT files found', subject, pCT_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'multiple pCT', False]], columns=error_log_columns), ignore_index=True)
            if len(brain_mask_files) < 1:
                print('No brain mask file found', subject, brain_mask_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'no brain mask', True]], columns=error_log_columns), ignore_index=True)
                continue
            if len(pCT_files) > 1:
                print('Multiple brain mask files found', subject, brain_mask_files)
                error_log_df = error_log_df.append(
                    pd.DataFrame([[subject, 'multiple brain mask', False]], columns=error_log_columns), ignore_index=True)

            # Check if already done
            if os.path.isfile(os.path.join(modality_dir, 'p_' + pCT_files[0] + '.gz')):
                print(subject, 'is already extracted. Skipping.')
                continue

            print('Extracting for', subject)
            # try:
            selected_pCT_file = os.path.join(modality_dir, pCT_files[0])
            selected_spc_file = os.path.join(modality_dir, spc_files[0])
            selected_brain_mask_file = os.path.join(modality_dir, brain_mask_files[0])

            temp_folder = os.path.join(modality_dir, 'temp_pct_preprocessing')
            if not os.path.exists(temp_folder):
                os.mkdir(temp_folder)

            # Motion correction
            motion_corrected_pCT = mcflirt(selected_pCT_file, outdir=temp_folder, verbose=True, stages=4)
            motion_corrected_pCT += '.gz'

            # Coregistration to non-contrast anatomical
            coregistered_pCT = coregistration_4D(motion_corrected_pCT, selected_spc_file)

            # Brain extraction
            output_path = os.path.join(modality_dir, 'p_' + pCT_files[0] + '.gz')
            brain_extraction(coregistered_pCT, output_path, brain_mask=selected_brain_mask_file)

            shutil.rmtree(temp_folder)
            # except Exception as e:
            #     print(e)
            #     error_log_df = error_log_df.append(
            #         pd.DataFrame([[subject, str(e), True]], columns=error_log_columns),
            #         ignore_index=True)

            error_log_df.to_excel(os.path.join(data_dir, 'pCT_preprocessing_error_log' + timestamp + '.xlsx'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion correct and align to native CT')
    parser.add_argument('input_directory')
    parser.add_argument("--reverse", nargs='?', const=True, default=False, help="Read directory in reverse.")
    args = parser.parse_args()
    pCT_preprocessing_pipeline(args.input_directory, args.reverse)
