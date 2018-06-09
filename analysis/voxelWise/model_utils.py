import os, timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
import receptiveField as rf
from cv_utils import balance, repeated_kfold_cv, intermittent_repeated_kfold_cv, ext_mem_repeated_kfold_cv

def create(model_dir, model_name, input_data_list, output_data_list, receptive_field_dimensions):
    rf_inputs, rf_outputs = rf.reshape_to_receptive_field(input_data_list, output_data_list, receptive_field_dimensions)

    # Create model object
    # model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000)
    # model = RandomForestClassifier(verbose = 1)
    model = XGBClassifier(verbose_eval=True)

    # Reduce amount of data initially processed
    remaining_fraction = 0.3
    print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
    X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 0.7, random_state = 42)

    # Balancing the data for training
    X_train, y_train = balance(X_retained, y_retained)


    # Train the model using the training sets
    model.fit(X_train, y_train)

    model_extension = '.pkl'
    model_path = os.path.join(model_dir, model_name + model_extension)
    print('Saving model as : ', model_path)
    joblib.dump(model, model_path)

    return model

def evaluate_crossValidation(save_dir, model_dir, model_name, input_data_array, output_data_array, receptive_field_dimensions):

    model = XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

    params = {
        'tree_method': 'hist',
        'silent': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }

    n_repeats = 1
    n_folds = 5

    results = ext_mem_repeated_kfold_cv(params, save_dir, input_data_array, output_data_array, receptive_field_dimensions, n_repeats, n_folds)
    # results = intermittent_repeated_kfold_cv(model, save_dir, input_data_array, output_data_array, receptive_field_dimensions, n_repeats, n_folds)
    # results = repeated_kfold_cv(model, input_data_array, output_data_array, receptive_field_dimensions, n_repeats, n_folds)

    accuracy = np.median(results['test_accuracy'])
    roc_auc = np.median(results['test_roc_auc'])
    f1 = np.median(results['test_f1'])

    print('Results for', model_name)
    print('Voxel-wise accuracy: ', accuracy)
    print('ROC AUC score: ', roc_auc)
    print('F1 score: ', f1)

    np.save(os.path.join(model_dir, model_name + '_' + str(receptive_field_dimensions[0]) + '_cv_scores.npy'), results)

    return accuracy, roc_auc, f1
