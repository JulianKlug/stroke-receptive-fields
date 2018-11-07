import sys, torch
sys.path.insert(0, '../')

import os, timeit
import nibabel as nib
import numpy as np
import receptiveField as rf
import visual, scoring_utils
import data_loader
from cv_framework import standardise


main_dir = '/Users/julian/master/data/from_Server'
data_dir = os.path.join(main_dir, '')
main_model_dir = os.path.join(main_dir, 'models')
model_extension = '.npy'

input_dir = os.path.join(data_dir, '')
input_image_path = os.path.join(input_dir, '316724/Ct2_Cerebrale/wcoreg_RAPID_MTT_1062561.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

model_name = 'test1_1'
rf = 1
feature_scaling = True

CLIN, IN, OUT, MASKS = data_loader.load_saved_data(data_dir)
CLIN = None

input_data = IN[0]
output_GT = OUT[0]

print('input shape', input_data.shape)

if feature_scaling == True:
    input_data, CLIN = standardise(input_data, CLIN)

rf_dim = [rf, rf, rf]

print('Predict output image with model: ', model_name)
model_dir = os.path.join(main_model_dir, model_name + '_output')
model_path = os.path.join(model_dir, 'trained_model_' + model_name + model_extension)
model = torch.load(model_path)

import xgboost as xgb
data = xgb.DMatrix(input_data.reshape(1, -1))
start = timeit.default_timer()
probas = model.predict(data,
            ntree_limit = model.best_ntree_limit)
end = timeit.default_timer()
print('Prediction time: ', end - start)

print('Predicted shape', probas.shape)

threshold = 0.5
lesion_volume = np.sum(probas > threshold)
print('Predicted lesion volume', lesion_volume.shape)

n_x, n_y, n_z, n_c = input_data.shape
results = scoring_utils.evaluate(probas, OUT[0], MASKS[0], 1, n_x, n_y, n_z)

coordinate_space = input_img.affine
image_extension = '.nii'
predicted_img = nib.Nifti1Image(probas, affine=coordinate_space)
nib.save(predicted_img, os.path.join(model_dir, model_name + '' + image_extension))


visual.display(probas)
