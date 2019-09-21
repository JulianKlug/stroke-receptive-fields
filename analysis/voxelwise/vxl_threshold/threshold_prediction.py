import sys, os
sys.path.insert(0, '../../'); sys.path.insert(0, '../')
import nibabel as nib
import numpy as np
from RAPID_model import RAPID_Model_Generator
from Tmax6 import Tmax6_Model_Generator
import data_loader, visual
import scoring_utils
from scipy.ndimage.morphology import binary_closing, binary_erosion, binary_dilation

def smooth_prediction(prediction):
    # Recover 3D shape of data
    smooth_pred = np.zeros(prediction.shape[:4])
    structure = np.ones((1, 1, 1), dtype=np.int)
    for subj in range(prediction.shape[0]):
        smooth_pred[subj] = binary_erosion(np.squeeze(prediction[subj]), structure)
        smooth_pred[subj] = binary_dilation(smooth_pred[subj], structure)
        smooth_pred[subj] = binary_closing(smooth_pred[subj], np.ones((4, 4, 4), dtype=np.int))
    # flat_smooth_pred = smooth_pred[data_positions == 1].reshape(-1)
    return smooth_pred

def rescale_outliers(imgX, MASKS):
    '''
    Rescale outliers as some images from RAPID seem to be scaled x10
    Outliers are detected if their median exceeds 5 times the global median and are rescaled by dividing through 10
    :param imgX: image data (n, x, y, z, c)
    :return: rescaled_imgX
    '''
    median_c0 = np.median(imgX[..., 0][MASKS])
    median_c1 = np.median(imgX[..., 1][MASKS])
    median_c2 = np.median(imgX[..., 2][MASKS])
    median_c3 = np.median(imgX[..., 3][MASKS])
    for i in range(imgX.shape[0]):
        if np.median(imgX[i, ..., 0][MASKS[i]]) > 5 * median_c0:
            imgX[i, ..., 0] = imgX[i, ..., 0] / 10
        if np.median(imgX[i, ..., 1][MASKS[1]]) > 5 * median_c1:
            imgX[i, ..., 1] = imgX[i, ..., 1] / 10
        if np.median(imgX[i, ..., 2][MASKS[i]]) > 5 * median_c2:
            imgX[i, ..., 2] = imgX[i, ..., 2] / 10
        if np.median(imgX[i, ..., 3][MASKS[i]]) > 5 * median_c3:
            imgX[i, ..., 3] = imgX[i, ..., 3] / 10

    return imgX


main_dir = '/Users/julian/master/data/all2016_subset_prepro'
data_dir = os.path.join(main_dir, '')

ref_dir = os.path.join(data_dir, '')
ref_image_path = os.path.join(ref_dir, 'subj-6ff0284e/pCT/wcoreg_CBF_subj-6ff0284e.nii')
ref_img = nib.load(ref_image_path)
subj_id = os.path.basename(os.path.dirname(os.path.dirname(ref_image_path)))
print('Predicting lesion for ', subj_id)


CLIN, IN, OUT, MASKS, IDS, PARAMS = data_loader.load_saved_data(data_dir)
IN = rescale_outliers(IN, MASKS)
subj_index = np.where(IDS == subj_id)[0]
IN, OUT, MASKS, IDS = IN[subj_index], OUT[subj_index], MASKS[subj_index], IDS[subj_index]

feature_scaling = False;
n_c = IN.shape[-1];
receptive_field_dimensions = 0;



Tmax6_Generator = Tmax6_Model_Generator(IN[..., 0].shape, feature_scaling)
Tmax6_model = Tmax6_Generator('', '', n_channels=1, n_channels_out=1, rf=receptive_field_dimensions)
CBF_Model_Generator = RAPID_Model_Generator(IN.shape, feature_scaling, threshold=0.6, post_smoothing=True)
CBF_model = CBF_Model_Generator('', '', n_channels=n_c, n_channels_out=1, rf=receptive_field_dimensions)

CBF_pred = CBF_model.predict(IN[MASKS], MASKS)
CBF_pred_img = np.zeros(MASKS.shape)
CBF_pred_img[MASKS] = CBF_pred
CBF_pred_img = np.squeeze(CBF_pred_img)

Tmax6_pred = Tmax6_model.predict(IN[MASKS][..., 0], MASKS)
Tmax6_pred_img = np.zeros(MASKS.shape)
Tmax6_pred_img[MASKS] = Tmax6_pred
# Tmax6_pred_img = smooth_prediction(Tmax6_pred_img)
Tmax6_pred_img = np.squeeze(Tmax6_pred_img)

# visual.display(pred_img)
coordinate_space = ref_img.affine
image_extension = '.nii'

CBF_predicted_img = nib.Nifti1Image(CBF_pred_img, affine=coordinate_space)
nib.save(CBF_predicted_img, os.path.join(data_dir, subj_id + '_CBF60%_prediction' + image_extension))

Tmax6_predicted_img = nib.Nifti1Image(Tmax6_pred_img, affine=coordinate_space)
nib.save(Tmax6_predicted_img, os.path.join(data_dir, subj_id + '_Tmax6_prediction' + image_extension))

threshold = 0.5
lesion_volume = np.sum(CBF_pred_img > threshold)
print('Predicted lesion volume', lesion_volume, 'voxels /', lesion_volume*0.008, 'ml' )

n_x, n_y, n_z = MASKS[0].shape
results = scoring_utils.evaluate(CBF_pred_img.reshape(-1), np.squeeze(OUT).reshape(-1), MASKS, [subj_id],  1, n_x, n_y, n_z)
accuracy = np.median(results['accuracy'])
roc_auc = np.median(results['roc_auc'])
f1 = np.median(results['f1'])
dice = np.median([item for item in results['image_wise_dice']])
hausdorff_distance = np.median([item for item in results['image_wise_hausdorff']])

print('Results for', subj_id)
print('Voxel-wise accuracy: ', accuracy)
print('ROC AUC score: ', roc_auc)
print('Dice score: ', dice)
print('Classic Hausdorff', hausdorff_distance)
print('F1 score: ', f1)


