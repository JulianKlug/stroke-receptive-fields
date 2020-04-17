import os, sys
import subprocess
sys.path.insert(0, '../')
from tools.segmentation.brain_extraction import brain_extraction

def align_FOV(image_to_bet, no_contrast_anatomical, data_dir):

    # Use robustfov (FSL) to get FOV on head only
    print('Setting FOV')
    cropped_path = os.path.join(data_dir, 'cropped_' + image_to_bet)
    subprocess.run(['robustfov', '-i', image_to_bet, '-r', cropped_path], cwd = data_dir)

    print('Coregistering to', no_contrast_anatomical)
    coreg_name = 'coreg_crp_' + image_to_bet + '.gz'
    coreg_path = os.path.join(data_dir, coreg_name)
    spc_path = os.path.join(data_dir, no_contrast_anatomical)
    subprocess.run([
        'flirt',
        '-in',  cropped_path,
        '-ref', spc_path, '-out', coreg_path, '-omat', os.path.join(data_dir, 'coreg.mat'),
        '-bins', '256', '-cost', 'mutualinfo', '-searchrx', '-90', '90', '-searchry', '-90', '90', '-searchrz', '-90', '90', '-dof', '12', '-interp', 'trilinear'
    ], cwd = data_dir)

def extract_brain(image_to_bet, no_contrast_anatomical, data_dir, brain_mask=True):
    output_path = os.path.join(data_dir, 'betted_' + image_to_bet)
    coreg_name = 'coreg_crp_' + image_to_bet + '.gz'
    coreg_path = os.path.join(data_dir, coreg_name)

    if not brain_mask:
        brain_extraction(coreg_path, output_path, no_contrast_anatomical=no_contrast_anatomical)
    else:
        brain_extraction(coreg_path, output_path, brain_mask=os.path.join(data_dir, 'brain_mask.nii'))

    # Todo Above brain_extraction tool is not yet tested in this setting, the code below is thus preserved for now
    # if not brain_mask:
    #     print('Removing skull of', no_contrast_anatomical)
    #     skull_strip_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skull_strip.sh')
    #     subprocess.run([skull_strip_path, '-i', no_contrast_anatomical], cwd = data_dir)
    #     mask_path = os.path.join(data_dir, 'betted_' + no_contrast_anatomical + '_Mask.nii.gz')
    # else:
    #     mask_path = os.path.join(data_dir, 'brain_mask.nii')
    #
    # print('Applying mask')
    # subprocess.run([
    #     'fslmaths', coreg_path, '-mas', mask_path, output_path
    # ], cwd = data_dir)

    print('Done with', image_to_bet)
