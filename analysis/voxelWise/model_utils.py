import sys
sys.path.insert(0, '../')

import os, timeit, torch
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import receptiveField as rf
from hyperopt import hp, fmin, rand, tpe, STATUS_OK, Trials
from cv_utils import repeated_kfold_cv, intermittent_repeated_kfold_cv, ext_mem_repeated_kfold_cv, external_evaluate_patient_wise_kfold_cv
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
    # model = xgb.XGBClassifier(verbose_eval=True, n_jobs = -1, tree_method = 'hist')

    params = {
        'n_jobs':-1,
        'base_score': 0.5,
        'booster': 'gbtree',
        'colsample_bylevel': 1,
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_delta_step': 0,
        'max_depth': 3,
        'min_child_weight': 1,
        'missing': None,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'reg_alpha': 0, 'reg_lambda': 1,
        'scale_pos_weight': 1,
        'seed': 0,
        'silent': 1,
        'subsample': 1,
        'verbose_eval': True,
        'tree_method': 'hist'
    }

    obj = None
    feval = None
    n_estimators = 100 # number of boosted tree
    evals = ()
    early_stopping_rounds = None
    evals_result = {}


    X, y = input_data_array, output_data_array
    ## Load everything at once
    rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X, y, receptive_field_dimensions)
    X_train, y_train = balance(rf_inputs, rf_outputs)
    tempX, tempY = X_train, y_train

    # Train the model using the training sets
    # RAW interface
    dtrain = xgb.DMatrix(tempX, tempY)
    print(dtrain.get_label().shape, dtrain.feature_names)
    model = xgb.train(params, dtrain, n_estimators,
                              evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=True, xgb_model=None)

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
        'eval_metric': 'auc',
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

    obj = None
    feval = None
    n_estimators = 100 # number of boosted tree
    evals = ()
    early_stopping_rounds = None
    evals_result = {}

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

    trained_model = xgb.train(params, dtrain, n_estimators,
                              evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=True, xgb_model=None)
    os.remove(os.path.join(data_dir, model_name + '.r0-1.cache'))
    os.remove(os.path.join(data_dir, model_name + '.r0-1.cache.row.page'))

    model_extension = '.pkl'
    model_path = os.path.join(model_dir, model_name + model_extension)
    print('Saving model as : ', model_path)
    joblib.dump(trained_model, model_path)

def evaluate_crossValidation(save_dir, model_dir, model_name, receptive_field_dimensions, data_dir = None, input_data_array = None, output_data_array = None, create_folds = True):

    # model = xgb.XGBClassifier(verbose_eval=False, n_jobs = -1, tree_method = 'hist')

    params = {
        'base_score': 0.5,
        'booster': 'gbtree',
        'colsample_bylevel': 1,
        'colsample_bytree': 1,
        'eval_metric': 'auc',
        'gamma': 0.84,
        'learning_rate': 0.4,
        'max_delta_step': 0,
        'max_depth': 1,
        'min_child_weight': 10.0,
        'n_estimators': 999,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'random_state': 0,
        'reg_alpha': 0.8671463346569078,
        'reg_lambda': 0.5916603334004378,
        'scale_pos_weight': 1,
        'silent': True,
        'subsample': 0.7925462136041614,
        'tree_method': 'hist'
    }


    n_repeats = 20
    n_folds = 5

    if create_folds:
        results = ext_mem_repeated_kfold_cv(params, save_dir, input_data_array, output_data_array, receptive_field_dimensions, n_repeats = n_repeats, n_folds = n_folds, create_folds = create_folds)
        # results = intermittent_repeated_kfold_cv(model, save_dir, input_data_array, output_data_array, receptive_field_dimensions, n_repeats, n_folds)
        # results = repeated_kfold_cv(model, input_data_array, output_data_array, receptive_field_dimensions, n_repeats, n_folds)
    else:
        results = external_evaluate_patient_wise_kfold_cv(params, data_dir)

    results['rf'] = receptive_field_dimensions

    accuracy = np.median(results['test_accuracy'])
    roc_auc = np.median(results['test_roc_auc'])
    f1 = np.median(results['test_f1'])

    print('Results for', model_name)
    print('Voxel-wise accuracy: ', accuracy)
    print('ROC AUC score: ', roc_auc)
    print('F1 score: ', f1)

    # save the results and the params objects
    torch.save(results, os.path.join(model_dir, 'scores_' + model_name + '.npy'))
    torch.save(params, os.path.join(model_dir, 'params_' + model_name + '.npy'))

    return accuracy, roc_auc, f1, params

class Hyperopt_objective():
    def __init__(self, data_dir, save_dir, input_data_array = None, output_data_array = None, receptive_field_dimensions = None, create_folds = False):
        super(Hyperopt_objective, self).__init__()
        self.data_dir = data_dir
        self.input_data_array = input_data_array
        self.output_data_array = output_data_array
        self.receptive_field_dimensions = receptive_field_dimensions
        self.create_folds = create_folds

        self.score_save_path = os.path.join(save_dir, 'custom_objective_best_score.npy')
        self.params_save_path = os.path.join(save_dir, 'custom_objective_best_params.npy')

        try:  # try to load an already saved trials object, and increase the max
            self.best_score = torch.load(self.score_save_path)
            self.best_params = torch.load(self.params_save_path)
        except:  # create a new trials object and start searching
            self.best_score = 0
            self.best_params = None

    def estimate_loss(self, space):
            params = {
                'tree_method': 'hist',
                'n_estimators':999,  # number of boosted tree
                'silent':True,
                'objective':"binary:logistic",
                'eval_metric': 'auc',
                'booster':'gbtree',
                'n_jobs':-1,
                'max_depth' : int(space['max_depth']),
                'learning_rate' : space['learning_rate'],
                'gamma': space['gamma'],
                'min_child_weight': space['min_child_weight'],
                'max_delta_step':0,
                'subsample': space['subsample'],
                'colsample_bytree': space['colsample_bytree'],
                'colsample_bylevel':1,
                'reg_alpha': space['reg_alpha'],
                'reg_lambda': space['reg_lambda'],
                'scale_pos_weight':1,
                'base_score':0.5,
                'random_state':0
            }

            n_repeats = 1
            n_folds = 5

            if self.create_folds:
                results = ext_mem_repeated_kfold_cv(params, self.data_dir, self.input_data_array, self.output_data_array, self.receptive_field_dimensions, n_repeats, n_folds)
            else:
                results = external_evaluate_patient_wise_kfold_cv(params, self.data_dir)

            roc_auc = np.median(results['test_roc_auc'])
            print('Using params:', params)
            print("ROC AUC SCORE:", roc_auc)

            if roc_auc > self.best_score:
                print('Better score:', roc_auc)
                print('Used params:', params)
                self.best_score = roc_auc
                self.best_params = params
                torch.save(roc_auc, self.score_save_path)
                torch.save(params, self.params_save_path)

            return {'loss': 1-roc_auc, 'status': STATUS_OK }


def xgb_hyperopt(data_dir, save_dir, receptive_field_dimensions, max_trials = 500, create_folds = False, input_data_array = None, output_data_array = None):
    trials_step = 10  # how many additional trials to do after loading saved trials. 1 = save after iteration
    initial_max_trials = 10  # initial current_max_trials. put something small to not have to wait
    current_max_trials = 0

    best_params_path = os.path.join(save_dir, 'rf_' + str(receptive_field_dimensions[0]) + '_hyperopt_best.npy')
    trials_path = os.path.join(save_dir, 'rf_' + str(receptive_field_dimensions[0]) + '_hyperopt_trials.npy')

    space = {
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 5, dtype=int)),
        # 'max_depth': hp.quniform('x_max_depth', 5, 30, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.79, 0.81),
        # 'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'learning_rate': hp.quniform('learning_rate', 0.325, 0.4, 0.01),
        'gamma': hp.quniform('gamma', 0.84, 0.86, 0.02),
        'reg_alpha': hp.uniform('reg_alpha', 0.8, 0.9),
        'reg_lambda': hp.uniform('reg_lambda', 0.5, 0.6),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.9, 2),
    }

    while current_max_trials < max_trials:
        try:  # try to load an already saved trials object, and increase the max
            trials = torch.load(trials_path)
            print("Found saved Trials! Loading...")
            current_max_trials = len(trials.trials) + trials_step
            print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
        except:  # create a new trials object and start searching
            current_max_trials = initial_max_trials
            trials = Trials()

        objective = Hyperopt_objective(data_dir, save_dir, input_data_array, output_data_array, receptive_field_dimensions, create_folds = create_folds)
        best = fmin(fn = objective.estimate_loss,
                    space = space,
                    algo = tpe.suggest,
                    max_evals = current_max_trials,
                    trials = trials)

        print("Best:", best)

        # save the trials object
        torch.save(trials, trials_path)
        # save the best params
        np.save(best_params_path, best)

    return best
