import sys
sys.path.insert(0, '../')

import os
import numpy as np
import model_utils
import visual
import data_loader
import manual_data

main_dir = '/home/snarduzz/Data'
data_dir = os.path.join(main_dir, 'To_Preprocess')
model_dir = '/home/klug/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Path to save the model to
model_name = 'temp.pkl'
model_path = os.path.join(model_dir, model_name)
if os.path.isfile(model_path):
    # file exists
    print('This model already exists: ', model_path)
    validation = input('Type `yes` if you wish to delete your previous model:\t')
    if (validation != 'yes'):
        raise ValueError('Model already exists. Choose another model name or delete current model')

# ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
# ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
ct_sequences = ['wcoreg_RAPID_TMax_[s]']
mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load(data_dir, ct_sequences, mri_sequences)

rf_dim = [1, 1, 1]

trained_model, X_test, y_test = model_utils.create(model_dir, model_name, IN, OUT, rf_dim)

model_utils.stats(trained_model, X_test, y_test)
