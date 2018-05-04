import sys
sys.path.insert(0, '../')

import os
import nibabel as nib
import numpy as np
from sklearn.externals import joblib
import receptiveField as rf
import visual
import data_loader

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'analysis_test2')

input_dir = os.path.join(data_dir, 'pt1/Ct2_Cerebral_20160103')

input_image_path = os.path.join(input_dir, 'wcoreg_RAPID_TMax_[s]_.nii')
input_img = nib.load(input_image_path)
# input_data = input_img.get_data()

# ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
ct_sequences = ['wcoreg_RAPID_TMax_[s]']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load(data_dir, ct_sequences, mri_sequences)

input_data = IN[0]

rf_dim = [1, 1, 1]

model_path = os.path.join(data_dir, '5pOverSampled_test.pkl')
model = joblib.load(model_path)
predicted = rf.predict(input_data, model, rf_dim)
print('Predicted shape', predicted.shape)
print('Predicted lesion size', np.sum(predicted))

coordinate_space = input_img.affine
predicted_img = nib.Nifti1Image(predicted, affine=coordinate_space)
nib.save(predicted_img, os.path.join(data_dir,'5pOverSampled_test.nii'))


visual.display(predicted)
