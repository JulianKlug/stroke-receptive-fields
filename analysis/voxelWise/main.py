import sys
sys.path.insert(0, '../')

import os
import nibabel as nib
import numpy as np
from sklearn.externals import joblib
import models
import visual
import data_loader

main_dir = '/home/snarduzz/Data'
data_dir = os.path.join(main_dir, 'To_Preprocess')
model_dir = '/home/klug/models'

# Path to save the model to
model_path = os.path.join(data_dir, 'model1.pkl')
if os.path.isfile(model_path):
    # file exists
    print('This model already exists: ', model_path)
    validation = input('Type `yes` if you wish to delete your previous model:\t')
    if (validation != 'yes'):
        raise ValueError('Model already exists. Choose another model name or delete current model')

ct_sequences = ['wcoreg_RAPID_TMax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
# ct_sequences = ['wcoreg_RAPID_TMax']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load(data_dir, ct_sequences, mri_sequences)

rf_dim = [1, 1, 1]

model = models.create(IN, OUT, rf_dim)

print('Saving model as : ', model_path)
joblib.dump(model, model_path)
