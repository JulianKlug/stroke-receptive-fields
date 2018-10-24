import sys
sys.path.insert(0, '../')

import os, timeit, traceback, torch
import numpy as np
import timeit
from vxl_xgboost import model_utils
from vxl_xgboost.external_mem_xgb import External_Memory_xgb
from vxl_xgboost.ram_xgb import Ram_xgb
from vxl_glm.glm_cv import glm_continuous_repeated_kfold_cv
import visual
import data_loader
import manual_data
from email_notification import NotificationSystem
from cv_framework import repeated_kfold_cv

# main_dir = '/Users/julian/master/server_output'
main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, 'saved_data')
model_dir = os.path.join(main_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

notification_system = NotificationSystem()

main_save_dir = os.path.join(main_dir, 'temp_data')

CLIN, IN, OUT = data_loader.load_saved_data(data_dir)
CLIN = None
# IN, OUT = data_loader.load_saved_data(data_dir)
# IN, OUT = manual_data.load(data_dir)

for rf in range(3):
    rf_dim = [rf, rf, rf]

    model_name = 'cv_framework_rf_hyperopt_' + str(rf)
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
        save_folds = False
        n_repeats = 20
        n_folds = 5

        Model_Generator = Ram_xgb
        results, trained_models = repeated_kfold_cv(Model_Generator, save_dir,
            input_data_array = IN, output_data_array = OUT, clinical_input_array = CLIN,
            receptive_field_dimensions = rf_dim, n_repeats = n_repeats, n_folds = n_folds, messaging = notification_system)
        # results, trained_models = model_utils.evaluate_crossValidation(save_dir, model_dir, model_name, rf_dim, n_repeats = 1, n_folds = 3,
        #                                     clinical_input_array = CLIN, input_data_array = IN, output_data_array = OUT, create_folds = True, save_folds = save_folds, messaging = notification_system)
        # results, trained_models = glm_continuous_repeated_kfold_cv(IN, OUT, rf_dim, n_repeats = 1, n_folds = 3, messaging = notification_system)
        # params = 0

        accuracy = np.median(results['test_accuracy'])
        roc_auc = np.median(results['test_roc_auc'])
        f1 = np.median(results['test_f1'])
        params = results['model_params']

        print('Results for', model_name)
        print('Voxel-wise accuracy: ', accuracy)
        print('ROC AUC score: ', roc_auc)
        print('F1 score: ', f1)

        # save the results and the params objects
        torch.save(results, os.path.join(model_dir, 'scores_' + model_name + '.npy'))
        torch.save(results['model_params'], os.path.join(model_dir, 'params_' + model_name + '.npy'))
        torch.save(trained_models, os.path.join(model_dir, 'trained_models_' + model_name + '.npy'))


        elapsed = timeit.default_timer() - start
        print('Evaluation done in: ', elapsed)
        title = model_name + ' finished Cross-Validation'
        body = 'accuracy ' + str(accuracy) + '\n' + 'ROC AUC ' + str(roc_auc) + '\n' + 'F1 ' + str(f1) + '\n' + 'RF ' + str(rf) + '\n' + 'Time elapsed ' + str(elapsed) + '\n' + str(params)
        notification_system.send_message(title, body)
    except Exception as e:
        title = model_name + ' errored upon rf_hyperopt'
        tb = traceback.format_exc()
        body = 'RF ' + str(rf) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
        notification_system.send_message(title, body)
        raise

print('Hyperopt done.')
