import sys
sys.path.insert(0, '../')

import os, timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import receptiveField as rf
from cv_utils import repeated_kfold_cv, intermittent_repeated_kfold_cv, ext_mem_repeated_kfold_cv
from ext_mem_utils import save_to_svmlight, delete_lines
from sampling_utils import get_undersample_selector_array, balance

def create(model_dir, model_name, input_data_array, output_data_array, receptive_field_dimensions):
    # Reduce amount of data initially processed
    # remaining_fraction = 0.3
    # print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
    # X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 0.7, random_state = 42)

    # Create model object
    # model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000)
    # model = RandomForestClassifier(verbose = 1)
    model = xgb.XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

    params = {
        'tree_method': 'hist',
        'max_depth' : 3,
        'learning_rate' : 0.1,
        'n_estimators':100,
        'silent':True,
        'objective':"binary:logistic",
        'booster':'gbtree',
        'n_jobs':-1,
        'gamma':0,
        'min_child_weight':1,
        'max_delta_step':0,
        'subsample':1,
        'colsample_bytree':1,
        'colsample_bylevel':1,
        'reg_alpha':0,
        'reg_lambda':1,
        'scale_pos_weight':1,
        'base_score':0.5,
        'random_state':0
    }



    ## Load everything at once
    # rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X, y, receptive_field_dimensions)
    # X_train, y_train = balance(rf_inputs, rf_outputs)

    ## Continous loading
    X, y = input_data_array, output_data_array
    X_train, y_train = [],[]
    print('initial shape', X.shape, y.shape)

    # Balancing the data for training
    selector = get_undersample_selector_array(y)
    for subject in range(X.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_train, subj_y_train = np.expand_dims(X[subject], axis=0), np.expand_dims(y[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

        subj_X_train, subj_y_train = rf_inputs[selector[subject].reshape(-1)], rf_outputs[selector[subject].reshape(-1)]

        print('subject shapes', subj_X_train.shape, subj_y_train.shape)
        X_train.append(subj_X_train)
        y_train.append(subj_y_train)

    tempX, tempY = np.concatenate(X_train), np.concatenate(y_train)


    # Train the model using the training sets
    # RAW interface
    dtrain = xgb.DMatrix(tempX, tempY)
    print(dtrain.get_label().shape, dtrain.feature_names)
    model = xgb.train(params, dtrain)

    # SKLEARN interface
    # model.fit(tempX, tempY)

    model_extension = '.pkl'
    model_path = os.path.join(model_dir, model_name + model_extension)
    print('Saving model as : ', model_path)
    joblib.dump(model, model_path)

    return model

def create_external_memory(model_dir, model_name, data_dir, input_data_array, output_data_array, receptive_field_dimensions):
    ext_mem_extension = '.txt'

    params = {
        'tree_method': 'hist',
        'max_depth' : 3,
        'learning_rate' : 0.1,
        'n_estimators':100,
        'silent':True,
        'objective':"binary:logistic",
        'booster':'gbtree',
        'n_jobs':1,
        'gamma':0,
        'min_child_weight':1,
        'max_delta_step':0,
        'subsample':1,
        'colsample_bytree':1,
        'colsample_bylevel':1,
        'reg_alpha':0,
        'reg_lambda':1,
        'scale_pos_weight':1,
        'base_score':0.5,
        'random_state':0
    }


    X, y = input_data_array, output_data_array
    print('initial shape', X.shape)

    # Get balancing selector respecting population wide distribution
    balancing_selector = get_undersample_selector_array(y)

    train_data_path = os.path.join(data_dir, model_name + '_train' + ext_mem_extension)
    if os.path.isfile(train_data_path):
        print('Data for this model already exists: ', train_data_path)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Model already exists. Choose another data name or delete current model')
        else:
            os.remove(train_data_path)

    for subject in range(X.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_train, subj_y_train = np.expand_dims(X[subject], axis=0), np.expand_dims(y[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

        # Balance by using predefined balancing_selector
        subj_X_train, subj_y_train = rf_inputs[balancing_selector[subject].reshape(-1)], rf_outputs[balancing_selector[subject].reshape(-1)]

        print('subject shapes', subj_X_train.shape, subj_y_train.shape)
        save_to_svmlight(subj_X_train, subj_y_train, train_data_path)

    cache_path = os.path.join(data_dir, model_name + '.cache')
    dtrain = xgb.DMatrix(train_data_path + '#' + cache_path)

    print('final shapes', dtrain.get_label().shape, dtrain.feature_names)

    trained_model = xgb.train(params, dtrain)
    os.remove(os.path.join(data_dir, model_name + '.r0-1.cache'))
    os.remove(os.path.join(data_dir, model_name + '.r0-1.cache.row.page'))

    model_extension = '.pkl'
    model_path = os.path.join(model_dir, model_name + model_extension)
    print('Saving model as : ', model_path)
    joblib.dump(trained_model, model_path)

def evaluate_crossValidation(save_dir, model_dir, model_name, input_data_array, output_data_array, receptive_field_dimensions):

    model = xgb.XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

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
