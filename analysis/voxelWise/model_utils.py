import os, timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from collections import Counter
import receptiveField as rf
from cv_utils import balance, repeated_kfold_cv, intermittent_repeated_kfold_cv, ext_mem_repeated_kfold_cv
from ext_mem_utils import save_to_svmlight, delete_lines

def create(model_dir, model_name, input_data_array, output_data_array, receptive_field_dimensions):
    # Create model object
    # model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000)
    # model = RandomForestClassifier(verbose = 1)
    # model = xgb.XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

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


    # rf_inputs, rf_outputs = rf.reshape_to_receptive_field(input_data_array, output_data_array, receptive_field_dimensions)

    # Reduce amount of data initially processed
    # remaining_fraction = 0.3
    # print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
    # X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 0.7, random_state = 42)

    # Balancing the data for training
    # X_train, y_train = balance(rf_inputs, rf_outputs)


    X, y = input_data_array, output_data_array
    X_train, y_train = [],[]
    print('initial shape', X.shape)
    for subject in range(X.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_train, subj_y_train = np.expand_dims(X[subject], axis=0), np.expand_dims(y[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)
        # TODO: Might be dangerous to balance here
        subj_X_train, subj_y_train = balance(rf_inputs, rf_outputs)
        print('subject shapes', subj_X_train.shape, subj_y_train.shape)
        X_train.append(subj_X_train)
        y_train.append(subj_y_train)

    tempX, tempY = np.concatenate(X_train), np.concatenate(y_train)
    print('ésdlifjsdàlgj', tempX.shape, tempY.shape)


    dtrain = xgb.DMatrix(tempX, tempY)

    print(dtrain.get_label().shape, dtrain.feature_names)

    model = xgb.train(params, dtrain)

    # Train the model using the training sets
    # model.fit(X_train, y_train)

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

    for subject in range(X.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_train, subj_y_train = np.expand_dims(X[subject], axis=0), np.expand_dims(y[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)
        # TODO: Might be dangerous to balance here
        # subj_X_train, subj_y_train = balance(rf_inputs, rf_outputs)
        subj_X_train, subj_y_train = rf_inputs, rf_outputs

        print('subject shapes', subj_X_train.shape, subj_y_train.shape)
        train_data_path = os.path.join(data_dir, model_name + '_train' + ext_mem_extension)
        save_to_svmlight(subj_X_train, subj_y_train, train_data_path)

    ext_mem_undersample(os.path.join(data_dir, model_name + '_train' + ext_mem_extension))

    # dtrain = xgb.DMatrix(os.path.join(data_dir, model_name + '_train' + ext_mem_extension)
    #     + '#' + os.path.join(data_dir, 'dtrain.cache'))
    dtrain = xgb.DMatrix(os.path.join(data_dir, model_name + '_train' + ext_mem_extension))

    print('final shapes', dtrain.get_label().shape, dtrain.feature_names)

    trained_model = xgb.train(params, dtrain)

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

def undersample(X, y):
    print('initial', X.shape, y.shape)
    undersampled_indices, unselected_indices = index_undersample_balance(y)
    return (X[undersampled_indices], y[undersampled_indices])
    # return (np.delete(X, unselected_indices, 0), np.delete(y, unselected_indices, 0))

def ext_mem_undersample(datapath):
    print('Undersampling data:', datapath)
    data = xgb.DMatrix(datapath)
    labels = data.get_label()
    undersampled_indices, unselected_indices = index_undersample_balance(labels)
    print('unselect lines', unselected_indices.shape)
    delete_lines(datapath, unselected_indices)

def index_undersample_balance(y):
    """
    Find indeces fitting a undersampled balance

    Args:
        y: dependent variables of data in a form of an np array (0,1 where 1 is underrepresented)

    Returns:
        undersampled_indices : indeces retained after random undersapling
        unselected_indices : indeces rejected after randum undersampling
    """
    print('Undesampling Ratio 1:1')
    n_pos = np.sum(y)
    neg_indices = np.squeeze(np.argwhere(y == 0))
    pos_indices = np.squeeze(np.argwhere(y == 1))
    randomly_downsampled_neg_indices = np.random.choice(neg_indices, int(n_pos), replace = False)
    undersampled_indices = np.concatenate([pos_indices, randomly_downsampled_neg_indices])
    unselected_indices = np.setdiff1d(neg_indices, randomly_downsampled_neg_indices)

    return (undersampled_indices, unselected_indices)
