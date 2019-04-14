import os, shutil

main_dir = '/Volumes/stroke_hdd1/stroke_db/2017'
data_dir = os.path.join(main_dir, 'extracted_data')

def flatten(data_dir):
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
            for study in studies:
                study_dir = os.path.join(modality_dir, study)
                files = [o for o in os.listdir(study_dir)]
                for file in files:
                    new_file_path = os.path.join(modality_dir, file)
                    if not os.path.exists(new_file_path):
                        shutil.move(os.path.join(study_dir, file), new_file_path)
                    else:
                        shutil.move(os.path.join(study_dir, file), os.path.join(modality_dir, study + '_' + file))
                shutil.rmtree(study_dir)

flatten(data_dir)
