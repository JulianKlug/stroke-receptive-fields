import os, argparse
import brain_extraction
import pandas as pd
import time

def brain_extract_wrapper(data_dir, reverse_reading):
    spc_start = 'SPC_301mm'
    # Find default Angio file, no MIP projection
    angio_start = 'Angio_CT_075'

    error_log_columns = ['subject', 'message', 'excluded']
    error_log_df = pd.DataFrame(columns=error_log_columns)

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    # allow for simultaneous processing
    if reverse_reading:
        subjects.reverse()

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)

            if modality.startswith('pCT'):
                spc_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(spc_start) and i.endswith('.nii')]
                angio_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(angio_start) and i.endswith('.nii')]

                if len(spc_files) < 1:
                    print('No SPC file found', subject, spc_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'no SPC', True]], columns=error_log_columns), ignore_index=True)
                    continue
                if len(spc_files) > 1:
                    print('Multiple SPC files found', subject, spc_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'multiple SPC', False]], columns=error_log_columns), ignore_index=True)
                if len(angio_files) < 1:
                    print('No Angio file found', subject, angio_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'no Angio', True]], columns=error_log_columns), ignore_index=True)
                    continue
                if len(angio_files) > 1:
                    print('Multiple Angio files found', subject, angio_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'multiple Angio', False]], columns=error_log_columns), ignore_index=True)

                # Check if already done
                if os.path.isfile(os.path.join(modality_dir, 'betted_' + angio_files[0] + '.gz')):
                    print(subject, 'is already extracted. Skipping.')
                    continue

                print('Extracting for', subject)
                brain_extraction.align_FOV(angio_files[0], spc_files[0], modality_dir)
                brain_extraction.extract_brain(angio_files[0], spc_files[0], modality_dir)

    error_log_df.to_excel(os.path.join(data_dir, 'brain_extraction_error_log' + str(time.time()).split('.')[0] + '.xlsx'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract brain region and align to native CT')
    parser.add_argument('input_directory')
    parser.add_argument("--reverse", nargs='?', const=True, default=False, help="Read directory in reverse.")
    args = parser.parse_args()
    brain_extract_wrapper(args.input_directory, args.reverse)

