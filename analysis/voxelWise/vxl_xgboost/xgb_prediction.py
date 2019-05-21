import sys, torch
sys.path.insert(0, '../')

import os, timeit
import nibabel as nib
import numpy as np
from receptiveField import reshape_to_receptive_field
import visual, scoring_utils
import data_loader
from cv_framework import standardise

main_dir = '/Users/julian/master/server_output'
data_dir = os.path.join(main_dir, 'LOO')
# main_model_dir = os.path.join(main_dir, 'models')
main_model_dir = '/Users/julian/master/server_output/'
model_extension = '.npy'

input_dir = os.path.join(data_dir, '')
input_image_path = os.path.join(input_dir, '448776/Ct2_Cerebrale/wcoreg_RAPID_MTT_448776.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

model_name = 'trained_std_ram_mask_xgb_rf_1'
rf = 1
feature_scaling = True

CLIN, IN, OUT, MASKS = data_loader.load_saved_data(data_dir)
CLIN = None

subj_index = 0
input_data = IN[subj_index]
output_GT = OUT[subj_index]
mask_data = MASKS[subj_index]
# MAKS = np.full(IN.shape, True)

print('input shape', input_data.shape)

if feature_scaling == True:
    input_data, CLIN = standardise(input_data, CLIN)

rf_dim = [rf, rf, rf]
n_x, n_y, n_z, n_c = input_data.shape

print('Predict output image with model: ', model_name)
model_dir = os.path.join(main_model_dir, model_name + '_output')
model_path = os.path.join(model_dir, 'trained_model_' + model_name + model_extension)
model = torch.load(model_path)

input_data, output_GT, masks = np.expand_dims(input_data, axis=0), np.expand_dims(output_GT, axis=0), np.expand_dims(mask_data, axis=0)
rf_inputs, rf_outputs = reshape_to_receptive_field(input_data, output_GT, rf_dim)
if CLIN is not None:
    # Add clinical data to every voxel
    mixed_inputs = np.zeros((rf_inputs.shape[0], rf_inputs.shape[1] + CLIN[0].shape[0]), np.float) # Initialising matrix of the right size
    mixed_inputs[:, : rf_inputs.shape[1]] = rf_inputs
    mixed_inputs[:, rf_inputs.shape[1] :]= CLIN[0]
    all_inputs = mixed_inputs
else:
    all_inputs = rf_inputs

X, y = all_inputs[masks.reshape(-1)], rf_outputs[masks.reshape(-1)]

import xgboost as xgb
data = xgb.DMatrix(X)
start = timeit.default_timer()
probas = model.predict(data,
            ntree_limit = model.best_ntree_limit)
end = timeit.default_timer()
print('Prediction time: ', end - start)
print('Predicted shape', probas.shape)

threshold = 0.5
lesion_volume = np.sum(probas > threshold)
print('Predicted lesion volume', lesion_volume.shape)

results = scoring_utils.evaluate(probas, y, masks, 1, n_x, n_y, n_z)
accuracy = np.median(results['accuracy'])
roc_auc = np.median(results['roc_auc'])
f1 = np.median(results['f1'])
dice = np.median([item for item in results['image_wise_dice']])

print('Results for', model_name)
print('Voxel-wise accuracy: ', accuracy)
print('ROC AUC score: ', roc_auc)
print('Dice score: ', dice)
print('F1 score: ', f1)

subj_3D_probas = np.full(masks[0].shape, 0, dtype = np.float64)
subj_3D_probas[masks[0]] = probas

coordinate_space = input_img.affine
image_extension = '.nii'
predicted_img = nib.Nifti1Image(subj_3D_probas, affine=coordinate_space)
nib.save(predicted_img, os.path.join(model_dir, model_name + '' + image_extension))
torch.save(results, os.path.join(model_dir, 'LOO_scores_' + model_name + '.npy'))

visual.display(subj_3D_probas)
visual.display(OUT[0])
