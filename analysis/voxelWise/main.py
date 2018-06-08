import sys
sys.path.insert(0, '../')

import os
import numpy as np
import model_utils
import visual
import data_loader
import manual_data
from email_notification import NotificationSystem

main_dir = '/home/klug/data/working_data'
data_dir = os.path.join(main_dir, 'saved_data')
model_dir = '/home/klug/models/patient_wise'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Path to save the model to
model_name = 'test_cv.pkl'
model_path = os.path.join(model_dir, model_name)
if os.path.isfile(model_path):
    # file exists
    print('This model already exists: ', model_path)
    validation = input('Type `yes` if you wish to delete your previous model:\t')
    if (validation != 'yes'):
        raise ValueError('Model already exists. Choose another model name or delete current model')

notification_system = NotificationSystem()

# ct_sequences = ['wcoreg_RAPID_TMax_[s]', 'wcoreg_RAPID_MTT_[s]', 'wcoreg_RAPID_CBV', 'wcoreg_RAPID_CBF']
# ct_sequences = ['wcoreg_RAPID_Tmax', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV', 'wcoreg_RAPID_rCBF']
# ct_sequences = ['wcoreg_RAPID_TMax_[s]']
# mri_sequences = ['wcoreg_VOI_lesion']

IN, OUT = data_loader.load_saved_data(data_dir)
# IN, OUT = manual_data.load(data_dir)

rf = 0
rf_dim = [rf, rf, rf]

# model_utils.patient_wise_kfold_data_split(data_dir, IN, OUT, rf_dim)
# model_utils.train_CV(data_dir)

score, roc_auc, f1 = model_utils.evaluate_model(data_dir, model_dir, model_name, IN, OUT, rf_dim)

# title = model_name + ' finished Cross-Validation'
# body = 'accuracy ' + str(score) + '\n' + 'ROC AUC ' + str(roc_auc) + '\n' + 'F1 ' + str(f1)
# notification_system.send_message(title, body)
