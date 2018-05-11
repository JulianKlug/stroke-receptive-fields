import sys
sys.path.insert(0, '../')

import os
import nibabel as nib
import numpy as np
from sklearn.externals import joblib
import receptiveField as rf
import visual
import data_loader

main_dir = '/Users/julian/master/server_output/test1'
data_dir = os.path.join(main_dir, 'LOO')
model_dir = os.path.join(main_dir, 'models')
model_path = os.path.join(model_dir, 'xgb_all_oversampled1.pkl')

input_dir = os.path.join(data_dir, '')

input_image_path = os.path.join(input_dir, '898729/Ct2_Cerebrale/wcoreg_RAPID_MTT_898729.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load(input_dir, ct_sequences, mri_sequences)

input_data = IN[0]

rf_dim = [1, 1, 1]

model = joblib.load(model_path)
predicted = rf.predict(input_data, model, rf_dim)
print('Predicted shape', predicted.shape)
print('Predicted lesion size', np.sum(predicted))

coordinate_space = input_img.affine
predicted_img = nib.Nifti1Image(predicted, affine=coordinate_space)
nib.save(predicted_img, os.path.join(data_dir,'xgb_all_oversampled1_loo.nii'))


visual.display(predicted)
