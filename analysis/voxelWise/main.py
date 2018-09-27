import os, timeit, traceback, torch
import numpy as np
import timeit
from vxl_xgboost import model_utils
import visual
import data_loader
import manual_data
from email_notification import NotificationSystem
from vxl_glm.glm_cv import glm_continuous_repeated_kfold_cv

# main_dir = '/Users/julian/master/data/hyperopt_test_LOO'
main_dir = '/home/jk/master/data/pure_LOO'
# main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, 'saved_data')
model_dir = os.path.join(main_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Path to save the model to
model_name = 'voxel_wise_metrics_test1'
model_path = os.path.join(model_dir, model_name + '.pkl')
if os.path.isfile(model_path):
    # file exists
    print('This model already exists: ', model_path)
    validation = input('Type `yes` if you wish to delete your previous model:\t')
    if (validation != 'yes'):
        raise ValueError('Model already exists. Choose another model name or delete current model')

notification_system = NotificationSystem()


IN, OUT = data_loader.load_saved_data(data_dir)
# IN, OUT = manual_data.load(data_dir)


rf = 1
rf_dim = [rf, rf, rf]
print('Evaluating', model_name, 'with rf:', rf_dim)

main_save_dir = os.path.join(main_dir, 'external_mem_data')
save_dir = os.path.join(main_save_dir, model_name + '_data')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

try:
    start = timeit.default_timer()
    save_folds = False
    results, trained_models = model_utils.evaluate_crossValidation(save_dir, model_dir, model_name, rf_dim,
                                        input_data_array = IN, output_data_array = OUT, create_folds = True, save_folds = save_folds, messaging = notification_system)
    # results = glm_continuous_repeated_kfold_cv(IN, OUT, rf_dim, n_repeats = 1, n_folds = 3, messaging = notification_system)
    # params = 0
    # score, roc_auc, f1 = model_utils.evaluate_crossValidation(save_dir, model_dir, model_name, rf_dim, IN, OUT)
    # score, roc_auc, f1 = glm_continuous_repeated_kfold_cv()
    # best = model_utils.xgb_hyperopt(data_dir, save_dir, rf_dim, create_folds = False)
    # model_utils.create(model_dir, model_name, IN, OUT, rf_dim)
    # model_utils.create_external_memory(model_dir, model_name, data_dir, IN, OUT, rf_dim)

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
