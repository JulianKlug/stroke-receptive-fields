import os
import subprocess

def extract_brain(image_to_bet, no_contrast_anatomical, data_dir):
    output_path = os.path.join(data_dir, 'betted_' + image_to_bet)

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

    print('Removing skull of', no_contrast_anatomical)
    skull_strip_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skull_strip.sh')
    subprocess.run([skull_strip_path, '-i', no_contrast_anatomical], cwd = data_dir)

    print('Applying mask')
    mask_path = os.path.join(data_dir, 'betted_' + no_contrast_anatomical + '_Mask.nii.gz')
    subprocess.run([
        'fslmaths', coreg_path, '-mas', mask_path, output_path
    ], cwd = data_dir)

    print('Done with', image_to_bet)
