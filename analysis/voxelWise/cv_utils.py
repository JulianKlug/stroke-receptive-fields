import os, timeit, shutil, traceback
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from collections import Counter
import receptiveField as rf
from ext_mem_utils import save_to_svmlight
from sampling_utils import balance, get_undersample_selector_array

def repeated_kfold_cv(model, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    """
    Patient wise Repeated KFold Crossvalidation
    Advantage over sklearns implementation: returns TP and FP rates (useful for plotting ROC curve)


    Args:
        model: model to crossvalidate (must implement sklearns interface)
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folfs in kfold (ie. k)

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """

    print('Crossvalidation: ' + str(n_folds) +' fold, repeated : ' + str(n_repeats))
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []

    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration += 1
        print('Crossvalidation: Running ' + str(iteration) + ' of a total of ' + str(n_repeats))

        f = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Evaluating split : ' + str(f))
            print('spit', X[train].shape, y[train].shape)

            start = timeit.default_timer()
            rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X[train], y[train], receptive_field_dimensions)
            end = timeit.default_timer()
            print('Reshaped to receptive fields in: ', end - start)

            X_train, y_train = balance(rf_inputs, rf_outputs)

            # Reduce amount of data initially processed
            remaining_fraction = 0.3
            print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
            X_retained, X_rest, y_retained, y_rest = train_test_split(rf_inputs, rf_outputs, test_size = 1-remaining_fraction, random_state = 42)

            fittedModel = model.fit(X_retained, y_retained)

            X_test, y_test = rf.reshape_to_receptive_field(X[test], y[test], receptive_field_dimensions)

            probas_= fittedModel.predict_proba(X_test)
            # Compute ROC curve, area under the curve, f1, and accuracy
            threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            accuracies.append(accuracy_score(y_test, probas_[:, 1] > threshold))
            f1_scores.append(f1_score(y_test, probas_[:, 1] > threshold))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(tpr)
            fprs.append(fpr)
            f += 1

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model': model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }

def ext_mem_repeated_kfold_cv(params, data_dir, imgX, y, receptive_field_dimensions, clinX = None, n_repeats = 1, n_folds = 5, create_folds = False, save_folds = True, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory to use for saving the intermittent states
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folds in kfold (ie. k)
        create_folds (option, dafault False): boolean, if the folds should be created anew
        save_folds (optional, default True): boolean, if the created folds should be saved
        messaging (optional, defaults to None): instance of notification_system used to report errors

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': params of the model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """
    print('Using external memory version for Crossvalidation')
    print('Using params:', params)

    if create_folds:
        if not save_folds:
            results = ext_mem_continuous_repeated_kfold_cv(params, data_dir, imgX, y, receptive_field_dimensions, clinX, n_repeats, n_folds, messaging)
            return results

        external_save_patient_wise_kfold_data_split(data_dir, imgX, y, receptive_field_dimensions, clinX, n_repeats, n_folds)

    results = external_evaluation_wrapper_patient_wise_kfold_cv(params, data_dir)

    return results

def external_save_patient_wise_kfold_data_split(save_dir, imgX, y, receptive_field_dimensions, clinX = None, n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function creates and saves k datafolds of n-iterations for crossvalidations
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory to use for saving the intermittent states
        imgX: image input data to validate for all subjects in form of an np array [subject, x, y, z, c]
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folds in kfold (ie. k)
        messaging (optional, defaults to None): instance of notification_system used to report errors

    Returns: undefined
    """
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
        print('This directory already exists: ', save_dir)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Model already exists. Choose another model name or delete current model')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    print('Saving repeated kfold data split to libsvm format', n_repeats, n_folds)

    ext_mem_extension = '.txt'
    start = timeit.default_timer()
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(imgX, y):
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))

            # Create this fold
            try:
                external_create_fold(fold_dir, fold, imgX, y, receptive_field_dimensions, train, test, clinX = clinX, ext_mem_extension = ext_mem_extension)
                pass
            except Exception as e:
                tb = traceback.format_exc()
                print('Creation of fold failed.')
                print(e)
                print(tb)
                if (messaging):
                    title = 'Minor error upon fold creation rf_hyperopt at ' + str(receptive_field_dimensions)
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            fold += 1

    end = timeit.default_timer()
    print('Created and saved splits in: ', end - start)

def external_evaluation_wrapper_patient_wise_kfold_cv(params, data_dir):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function evaluates the saved datafolds.
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory of the saved iterations

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': params of the model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """
    print('Evaluating model with data on external memory')
    ext_mem_extension = '.txt'
    n_folds = 0;
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []

    iterations = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o)) and o.startswith('iteration')]
    n_repeats = len(iterations)

    for iteration in iterations:
        print(iteration)
        iteration_dir = os.path.join(data_dir, iteration)

        folds = [o for o in os.listdir(iteration_dir)
                        if os.path.isdir(os.path.join(iteration_dir,o)) and o.startswith('fold')]
        n_folds = len(folds)

        for fold in folds:
            print('Training', fold, 'of', iteration)
            fold_dir = os.path.join(iteration_dir, fold)

            fold_result = external_evaluate_fold_cv(params, fold_dir, fold, ext_mem_extension)
            accuracies.append(fold_result['accuracy'])
            f1_scores.append(fold_result['f1'])
            aucs.append(fold_result['roc_auc'])
            tprs.append(fold_result['TPR'])
            fprs.append(fold_result['FPR'])
            trained_model = fold_result['trained_model']

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model_params': params,
        'trained_model': trained_model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }

def ext_mem_continuous_repeated_kfold_cv(params, save_dir, imgX, y, receptive_field_dimensions, clinX = None, n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function creates and evaluates k datafolds of n-iterations for crossvalidation,
    BUT erases the saved data after every fold evaluation
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        data_dir: directory to use for saving the intermittent states
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
        imgX: image input data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folds in kfold (ie. k)
        messaging (optional, defaults to None): instance of notification_system used to report errors


    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': params of the model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """
    print('CONTINOUS REPEATED KFOLD CV')
    print('Attention! Folds will not be saved.')

    # Initialising variables for evaluation
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []
    failed_folds = 0

    if clinX is not None:
        print('Using clinical data.')
        if clinX.shape[0] != imgX.shape[0]:
            raise ValueError('Not the same number of clinical and imaging data points:', clinX.shape, imgX.shape)

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
        print('This directory already exists: ', save_dir)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Model already exists. Choose another model name or delete current model')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    print('Saving repeated kfold data split to libsvm format', n_repeats, n_folds)

    ext_mem_extension = '.txt'
    start = timeit.default_timer()
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(imgX, y):
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            # Create this fold
            try:
                external_create_fold(fold_dir, fold, imgX, y, receptive_field_dimensions, train, test, clinX = clinX, ext_mem_extension = ext_mem_extension)
                pass
            except Exception as e:
                tb = traceback.format_exc()
                print('Creation of fold failed.')
                print(e)
                print(tb)
                if (messaging):
                    title = 'Minor error upon fold creation rf_hyperopt at ' + str(receptive_field_dimensions)
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            # Evaluate this fold
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration))
            try:
                fold_result = external_evaluate_fold_cv(params, fold_dir, 'fold_' + str(fold), ext_mem_extension)
                accuracies.append(fold_result['accuracy'])
                f1_scores.append(fold_result['f1'])
                aucs.append(fold_result['roc_auc'])
                tprs.append(fold_result['TPR'])
                fprs.append(fold_result['FPR'])
                trained_model = fold_result['trained_model']
                pass
            except Exception as e:
                failed_folds += 1
                print('Evaluation of fold failed.')
                print(e)
                if (messaging):
                    title = 'Minor error upon fold evaluation rf_hyperopt at ' + str(receptive_field_dimensions)
                    tb = traceback.format_exc()
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            # Erase saved fold to free up space
            try:
                shutil.rmtree(fold_dir)
            except:
                print('No fold to clear.')

            fold += 1
            # End of fold iteration

        try:
            shutil.rmtree(iteration_dir)
        except:
            print('No iteration to clear.')
        # End of iteration iteration

    end = timeit.default_timer()
    print('Created, saved and evaluated splits in: ', end - start)

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'failed_folds': failed_folds,
        'model_params': params,
        'trained_model': trained_model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }

def external_create_fold(fold_dir, fold, imgX, y, receptive_field_dimensions, train, test, clinX = None, ext_mem_extension = '.txt'):
    """
    Create a fold given the data and the test / train distribution
    External Memory: saves the folds as libsvm files

    Args:
        fold_dir: directory to use for saving the fold
        fold : name of the fold
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
        imgX: image input data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        train: boolean array selecting for Training
        test: boolean array selecting for testing

    Returns: undefined
    """
    print('Creating fold : ' + str(fold))
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    imgX_train, y_train = imgX[train], y[train]
    if clinX is not None:
        clinX_train = clinX[train]

    # Get balancing selector respecting population wide distribution
    balancing_selector = get_undersample_selector_array(y_train)

    for subject in range(imgX_train.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_train, subj_y_train = np.expand_dims(imgX_train[subject], axis=0), np.expand_dims(y_train[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

        if clinX is not None:
            # Add clinical data to every voxel
            # As discussed here: https://stackoverflow.com/questions/52132331/how-to-add-multiple-extra-columns-to-a-numpy-array/52132400#52132400
            subj_mixed_inputs = np.zeros((rf_inputs.shape[0], rf_inputs.shape[1] + clinX_train[subject].shape[0]), dtype = np.float) # Initialising matrix of the right size
            subj_mixed_inputs[:, : rf_inputs.shape[1]] = rf_inputs
            subj_mixed_inputs[:, rf_inputs.shape[1] :]= clinX_train[subject]
            all_inputs = subj_mixed_inputs
        else:
            all_inputs = rf_inputs

        # Balance by using predefined balancing_selector
        subj_X_train, subj_y_train = all_inputs[balancing_selector[subject].reshape(-1)], rf_outputs[balancing_selector[subject].reshape(-1)]

        train_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_train' + ext_mem_extension)
        save_to_svmlight(subj_X_train, subj_y_train, train_data_path)

    X_test, y_test = imgX[test], y[test]
    if clinX is not None:
        clinX_test = clinX[test]
    for subject in range(X_test.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_test, subj_y_test = np.expand_dims(X_test[subject], axis=0), np.expand_dims(y_test[subject], axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_test, subj_y_test, receptive_field_dimensions)
        if clinX is not None:
            # Add clinical data to every voxel
            subj_mixed_inputs = np.zeros((rf_inputs.shape[0], rf_inputs.shape[1] + clinX_test[subject].shape[0]), np.float) # Initialising matrix of the right size
            subj_mixed_inputs[:, : rf_inputs.shape[1]] = rf_inputs
            subj_mixed_inputs[:, rf_inputs.shape[1] :]= clinX_test[subject]
            all_inputs = subj_mixed_inputs
        else:
            all_inputs = rf_inputs
        test_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_test' + ext_mem_extension)

        save_to_svmlight(all_inputs, rf_outputs, test_data_path)


def external_evaluate_fold_cv(params, fold_dir, fold, ext_mem_extension):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    This function evaluates a saved datafold
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate, needs to have n_estimators
        fold_dir: directory of the saved fold

    Returns: result dictionary
    """
    n_estimators = params['n_estimators']

    dtrain = xgb.DMatrix(os.path.join(fold_dir, fold + '_train' + ext_mem_extension)
        + '#' + os.path.join(fold_dir, 'dtrain.cache'))
    dtest = xgb.DMatrix(os.path.join(fold_dir, fold + '_test' + ext_mem_extension)
        + '#' + os.path.join(fold_dir, 'dtest.cache'))

    trained_model = xgb.train(params, dtrain,
        num_boost_round = n_estimators,
        evals = [(dtest, 'Test')],
        early_stopping_rounds = 30,
        verbose_eval = False)

    # Clean up cache files
    try:
        os.remove(os.path.join(fold_dir, 'dtrain.r0-1.cache'))
        os.remove(os.path.join(fold_dir, 'dtrain.r0-1.cache.row.page'))
        os.remove(os.path.join(fold_dir, 'dtest.r0-1.cache'))
        os.remove(os.path.join(fold_dir, 'dtest.r0-1.cache.row.page'))
    except:
        print('No cache to clear.')

    print('Testing', fold, 'in', fold_dir)
    y_test = dtest.get_label()
    probas_= trained_model.predict(dtest, ntree_limit = trained_model.best_ntree_limit)
    # Compute ROC curve, area under the curve, f1, and accuracy
    threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:])
    accuracy = accuracy_score(y_test, probas_[:] > threshold)
    f1 = f1_score(y_test, probas_[:] > threshold)
    roc_auc = auc(fpr, tpr)

    return {
        'trained_model': trained_model,
        'FPR': fpr,
        'TPR': tpr,
        'thresholds': thresholds,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
        }

def intermittent_repeated_kfold_cv(model, save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    """
    Patient wise Repeated KFold Crossvalidation
    Intermittency: saves the folds and the progressively loads them to save on RAM
    Advantage over sklearns implementation: returns TP and FP rates (useful for plotting ROC curve)


    Args:
        model: model to crossvalidate (must implement sklearns interface)
        save_dir: directory to use for saving the intermittent states
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folfs in kfold (ie. k)

    Returns: result dictionary
        'settings_repeats': n_repeats
        'settings_folds': n_folds
        'model': model that was evaluated
        'test_accuracy': accuracy in every fold of every iteration
        'test_roc_auc': auc of roc in every fold of every iteration
        'test_f1': f1 score in every fold of every iteration
        'test_TPR': true positive rate in every fold of every iteration
        'test_FPR': false positive rate in every fold of every iteration
    """

    save_patient_wise_kfold_data_split(save_dir, X, y, receptive_field_dimensions, n_repeats, n_folds)

    train_patient_wise_kfold_cv(model, save_dir)

    results = test_patient_wise_kfold_cv(save_dir, receptive_field_dimensions)

    return results


def save_patient_wise_kfold_data_split(save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Creating kfold data split')

    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        f = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Creating split : ' + str(f))

            start = timeit.default_timer()
            rf_inputs, rf_outputs = rf.reshape_to_receptive_field(X[train], y[train], receptive_field_dimensions)
            end = timeit.default_timer()
            print('Reshaped to receptive fields in: ', end - start)

            X_train, y_train = balance(rf_inputs, rf_outputs)
            X_test, y_test = X[test], y[test]
            np.savez_compressed(os.path.join(iteration_dir, 'fold_' + str(f)),
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )

            f += 1

def train_patient_wise_kfold_cv(model, data_dir):

    iterations = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o)) and o.startswith('iteration')]

    for iteration in iterations:
        print(iteration)
        iteration_dir = os.path.join(data_dir, iteration)

        folds = [o for o in os.listdir(iteration_dir)
                        if o.startswith('fold')]
        for fold in folds:
            print('Training', fold, 'of', iteration)
            X_train = np.load(os.path.join(iteration_dir, fold))['X_train']
            y_train = np.load(os.path.join(iteration_dir, fold))['y_train']

            # Reduce amount of data initially processed
            remaining_fraction = 0.3
            print('Discarding ' + str((1 - remaining_fraction)* 100) + '% of data for faster training')
            X_retained, X_rest, y_retained, y_rest = train_test_split(X_train, y_train, test_size = 1-remaining_fraction, random_state = 42)

            trained_model = model.fit(X_retained, y_retained)

            fold_name_pure = fold.split('.')[0]
            model_extension = '.pkl'
            model_path = os.path.join(iteration_dir, 'model_' + fold_name_pure + model_extension)
            print('Saving model as : ', model_path)
            joblib.dump(trained_model, model_path)

def test_patient_wise_kfold_cv(data_dir, receptive_field_dimensions):
    print('Testing Crossvalidation')
    n_folds = 0;
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []

    iterations = [o for o in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir,o)) and o.startswith('iteration')]
    n_repeats = len(iterations)

    for iteration in iterations:
        print(iteration)
        iteration_dir = os.path.join(data_dir, iteration)

        folds = [o for o in os.listdir(iteration_dir)
                        if o.startswith('fold')]
        n_folds = len(folds)
        for fold in folds:
            print('Testing', fold, 'of', iteration)
            X_test = np.load(os.path.join(iteration_dir, fold))['X_test']
            y_test = np.load(os.path.join(iteration_dir, fold))['y_test']
            fold_name_pure = fold.split('.')[0]
            model_extension = '.pkl'
            model = joblib.load(os.path.join(iteration_dir, 'model_' + fold_name_pure + model_extension))

            X_test, y_test = rf.reshape_to_receptive_field(X_test, y_test, receptive_field_dimensions)

            probas_= model.predict_proba(X_test)
            # Compute ROC curve, area under the curve, f1, and accuracy
            threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            accuracies.append(accuracy_score(y_test, probas_[:, 1] > threshold))
            f1_scores.append(f1_score(y_test, probas_[:, 1] > threshold))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(tpr)
            fprs.append(fpr)

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model': model,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
    }
