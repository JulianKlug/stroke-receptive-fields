import os, pydicom
from shutil import copytree

"""
Given a main directory, check if any subjects has a study named "RAPID37" that contains RAPID files
and copy them into the appropriate directory
"""

def extract_RAPID37_folder(data_dir):
    upper_modality_dir = os.path.dirname(data_dir)
    folders = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o))]

    for folder in folders:
        folder_dir = os.path.join(data_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]
        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            series = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]
            for serie in series:
                new_serie_path = os.path.join(upper_modality_dir, 'RAPID37' + '_' + serie)
                if not os.path.exists(new_serie_path):
                    copytree(os.path.join(modality_dir, serie), new_serie_path)

def extract_RAPID37_folders_wrapper(main_dir):
    folders = [o for o in os.listdir(main_dir)
                    if os.path.isdir(os.path.join(main_dir,o))]
    i = 0
    for folder in folders:
        folder_dir = os.path.join(main_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o))]
        for modality in modalities:
            modality_dir = os.path.join(folder_dir, modality)
            series = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o)) and o.startswith('RAPID_37')]
            for serie in series:
                print(folder, i / len(folders))
                extract_RAPID37_folder(os.path.join(modality_dir, serie))
        i = i + 1

data_dir = '/Volumes/stroke_hdd1/stroke_db/2016/part1'
extract_RAPID37_folders_wrapper(data_dir)
