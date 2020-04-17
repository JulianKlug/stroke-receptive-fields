import os, subprocess

def brain_extraction(image_to_bet, output_path, no_contrast_anatomical=None, brain_mask=None):
    '''
    Brain extraction for CT images.
    Apply either a pre-established brain mask or create a brain mask from a non contrast anatomical file and then apply
    :param image_to_bet: path to input image
    :param output_path: path of betted output
    :param no_contrast_anatomical: path to non-contrast image
    :param brain_mask: path to brain mask
    :return: output_path
    '''
    if brain_mask is None and no_contrast_anatomical is None:
        raise Exception('Please provide either a brain mask or a non-contrast anatomical file')

    data_dir = os.path.dirname(image_to_bet)

    if brain_mask is None:
        print('Removing skull of', no_contrast_anatomical)
        skull_strip_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'skull_strip.sh')
        subprocess.run([skull_strip_path, '-i', no_contrast_anatomical], cwd = data_dir)
        brain_mask = os.path.join(data_dir, 'betted_' + no_contrast_anatomical + '_Mask.nii.gz')

    print('Applying mask')
    subprocess.run([
        'fslmaths', image_to_bet, '-mas', brain_mask, output_path
    ], cwd = data_dir)

    return output_path

