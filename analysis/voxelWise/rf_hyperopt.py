import sys
sys.path.insert(0, '../')

import os, timeit, traceback
import numpy as np
import timeit
import model_utils
import visual
import data_loader
import manual_data
from email_notification import NotificationSystem

#main_dir = '/Users/julian/master/data/hyperopt_test_LOO'
main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, 'saved_data')
model_dir = os.path.join(main_dir, 'models/rf_repeat1')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

notification_system = NotificationSystem()

main_save_dir = os.path.join(main_dir, 'repeat1_rf_hyperopt_data')

IN, OUT = data_loader.load_saved_data(data_dir)
# IN, OUT = manual_data.load(data_dir)

for rf in range(3):
    rf_dim = [rf, rf, rf]

    model_name = 'repeat1_rf_hyperopt_' + str(rf)
    model_path = os.path.join(model_dir, model_name + '.pkl')
    if os.path.isfile(model_path):
        # file exists
        print('This model already exists: ', model_path)
        validation = input('Type `yes` if you wish to delete your previous model:\t')
        if (validation != 'yes'):
            raise ValueError('Model already exists. Choose another model name or delete current model')

    print('Evaluating', model_name, 'with rf:', rf_dim)

    save_dir = os.path.join(main_save_dir, model_name + '_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        start = timeit.default_timer()
        score, roc_auc, f1, params = model_utils.evaluate_crossValidation(save_dir, model_dir, model_name, input_data_array = IN, output_data_array = OUT, receptive_field_dimensions = rf_dim, create_folds = True)
        elapsed = timeit.default_timer() - start
        print('Evaluation done in: ', elapsed)
        title = model_name + ' finished Cross-Validation'
        body = 'accuracy ' + str(score) + '\n' + 'ROC AUC ' + str(roc_auc) + '\n' + 'F1 ' + str(f1) + '\n' + 'RF ' + str(rf) + '\n' + 'Time elapsed ' + str(elapsed) + '\n' + str(params)
        notification_system.send_message(title, body)
    except Exception as e:
        title = model_name + ' errored upon rf_hyperopt'
        tb = traceback.format_exc()
        body = 'RF ' + str(rf) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
        notification_system.send_message(title, body)
        raise

print('Hyperopt done.')
