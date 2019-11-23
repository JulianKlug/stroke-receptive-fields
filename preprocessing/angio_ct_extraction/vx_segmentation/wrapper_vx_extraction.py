import os, argparse, time, subprocess
import pandas as pd
import numpy as np
import nibabel as nib
from hessian_vessels import segment_hessian_vessels

extract_vx_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extract_vx.sh')


def wrapper_vx_extraction(data_dir):
    spc_start = 'SPC_301mm'
    # Find default Angio file, no MIP projection
    angio_start = 'Angio_CT_075'
    timestamp = str(time.time()).split('.')[0]

    subjects = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    error_log_columns = ['subject', 'message', 'excluded']
    error_log_df = pd.DataFrame(columns=error_log_columns)

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)

            if modality.startswith('pCT'):
                spc_files = [i for i in os.listdir(modality_dir) if os.path.isfile(os.path.join(modality_dir, i)) and i.startswith(spc_start) and i.endswith('.nii')]
                angio_files = [i for i in os.listdir(modality_dir)
                                    if os.path.isfile(os.path.join(modality_dir, i))
                                        and i.startswith('betted_' + angio_start) and i.endswith('.nii.gz')]

                if len(angio_files) < 1:
                    print('No Angio file found', subject, angio_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'no Angio', True]], columns=error_log_columns), ignore_index=True)
                    continue
                if len(angio_files) > 1:
                    print('Multiple Angio files found', subject, angio_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'multiple Angio', False]], columns=error_log_columns), ignore_index=True)
                if len(spc_files) < 1:
                    print('No SPC file found', subject, spc_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'no SPC', True]], columns=error_log_columns), ignore_index=True)
                    continue
                if len(spc_files) > 1:
                    print('Multiple SPC files found', subject, spc_files)
                    error_log_df = error_log_df.append(
                        pd.DataFrame([[subject, 'multiple SPC', False]], columns=error_log_columns), ignore_index=True)

                mask_path = os.path.join(modality_dir, 'brain_mask.nii')

                # Check if already done
                if os.path.isfile(os.path.join(modality_dir, 'extracted_' + angio_files[0])):
                    print(subject, 'vessels are already extracted. Skipping.')
                    continue

                print('Extracting Vessels for', subject)

                # vessel enhancement: 1. Subtract native CT + 2. Multiply by brain_mask
                vessel_enhancement(os.path.join(modality_dir, angio_files[0]),
                                   os.path.join(modality_dir, spc_files[0]),
                                   mask_path,
                                   os.path.join(modality_dir, 'enhanced_' + angio_files[0]))

                # # Newer hessian tubular vessel filter
                segment_hessian_vessels(os.path.join(modality_dir, 'enhanced_' + angio_files[0]),
                                        os.path.join(modality_dir, 'enhanced_extracted_' + angio_files[0]))
                segment_hessian_vessels(os.path.join(modality_dir, angio_files[0]),
                                        os.path.join(modality_dir, 'extracted_' + angio_files[0]))

                dual_filter(os.path.join(modality_dir, 'enhanced_extracted_' + angio_files[0]),
                            os.path.join(modality_dir, 'enhanced_' + angio_files[0]),
                            os.path.join(modality_dir, 'extracted_' + angio_files[0]),
                            os.path.join(modality_dir, angio_files[0]),
                            os.path.join(modality_dir, 'filtered_extracted_' + angio_files[0]))


                # # Old thresholding extraction process
                # subprocess.run([extract_vx_path, '-i', angio_files[0]], cwd = modality_dir)

                error_log_df.to_excel(os.path.join(data_dir, 'vessel_extraction_error_log' + timestamp + '.xlsx'))

def vessel_enhancement(angio_file, spc_file, brain_mask, output_file_path):
    #  todo describe
    print('Enhancing vessels')
    # Substract SPC to lower signal from non injected structures
    subprocess.run([
        'fslmaths', angio_file, '-sub', spc_file, output_file_path
    ], cwd=os.path.dirname(angio_file))
    # Mask image (again) to remove regions from SPC outside of brain
    subprocess.run([
        'fslmaths', output_file_path, '-mas', brain_mask, output_file_path
    ], cwd=os.path.dirname(angio_file))

def dual_filter(enhanced_vessels_file, enhanced_angio_file, vessels_file, angio_file, output_file_path):
    # Filter enhanced and non enhanced by a 5% intensity threshold
    # and keep only if both thresholds are passed
    # @return enhanced angio

    print('Filtering vessels at 5% intensity of enhanced and non enhanced image')
    filtered_enhanced_vessels, coordinate_space = intensity_filter(enhanced_vessels_file, enhanced_angio_file)
    filtered_non_enhanced_vessels, _ = intensity_filter(vessels_file, angio_file)

    data = filtered_enhanced_vessels
    data[filtered_non_enhanced_vessels == 0] = 0

    filtered_img = nib.Nifti1Image(data.astype('float64'), affine=coordinate_space)
    nib.save(filtered_img, output_file_path)

    # create mask
    data[data > 0] = 1
    filtered_mask = nib.Nifti1Image(data.astype('float64'), affine=coordinate_space)
    nib.save(filtered_mask, os.path.join(os.path.dirname(output_file_path), 'mask_' + os.path.basename(output_file_path)))
    return data

def intensity_filter(extracted_angio_file_path, angio_file_path):
    # Intensity filter filtering out values below 5% of normalised signal
    vessel_img = nib.load(extracted_angio_file_path)
    vessel_data = vessel_img.get_data()
    angio_img = nib.load(angio_file_path)
    angio_data = angio_img.get_data()

    # Weigh mask according to tracer signal
    data = np.multiply(vessel_data, angio_data)

    data[data <= 0] = 0
    # disregard extremes
    data = np.clip(data, 0, np.percentile(data[data > 0], 99))
    # normalise
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # threshold at 5% of normalised data
    data[data <= 0.05] = 0

    coordinate_space = vessel_img.affine
    return data, coordinate_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vessels (signal enhancement + hessian vesselness segmentation)')
    parser.add_argument('input_directory')
    args = parser.parse_args()
    wrapper_vx_extraction(args.input_directory)

