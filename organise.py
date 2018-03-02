import os
import subprocess

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'extracted_data')
output_dir = os.path.join(main_dir, 'preprocessing')
ct_sequences = ['SPC_301mm_Std', 'RAPID_TMax_[s]', 'RAPID_MTT_[s]', 'RAPID_CBV', 'RAPID_CBF']
mri_sequences = ['t2_tse_tra', 'T2W_TSE_tra']
sequences = ct_sequences + mri_sequences

subjects = os.listdir(data_dir)

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    if os.path.isdir(subject_dir):
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            modality_output_dir = os.path.join(output_dir, subject, modality)

            if not os.path.exists(modality_output_dir):
                os.makedirs(modality_output_dir)


            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]


            for study in studies:
                study_dir = os.path.join(modality_dir, study)
                if study in sequences:
                    for file in os.listdir(study_dir):
                        if file.endswith(".nii"):
                            file_path = os.path.join(study_dir, file)
                            new_file_name = study + '_' + subject + '.nii'
                            new_file_path = os.path.join(modality_output_dir, new_file_name)
                            if not os.path.exists(new_file_path):
                                subprocess.run(['cp', file_path, new_file_path])

        # copy lesions file into subject dir
        lesion_path = os.path.join(main_dir, 'working_data', subject, 'VOI lesion.nii')
        new_lesion_path = os.path.join(output_dir, subject, 'VOI_lesion.nii')
        if not os.path.exists(new_lesion_path):
            # print(new_lesion_path)
            subprocess.run(['cp', lesion_path, new_lesion_path])
