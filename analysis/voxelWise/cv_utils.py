import os, timeit
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

def ext_mem_repeated_kfold_cv(params, save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    """
    Patient wise Repeated KFold Crossvalidation for xgboost
    External Memory: saves the folds as libsvm files and uses the external memory version of xgboost to avoid overloading the RAM

    Args:
        params: params of the xgboost model to crossvalidate
        save_dir: directory to use for saving the intermittent states
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        n_repeats (optional, default 1): repeats of kfold CV
        n_folds (optional, default 5): number of folfs in kfold (ie. k)

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

    external_save_patient_wise_kfold_data_split(save_dir, X, y, receptive_field_dimensions, n_repeats, n_folds)

    results = external_evaluate_patient_wise_kfold_cv(params, save_dir)

    return results

def external_save_patient_wise_kfold_data_split(save_dir, X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5):
    if not os.path.exists(save_dir):
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
        for train, test in kf.split(X, y):
            print('Creating fold : ' + str(fold))
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            X_train, y_train = X[train], y[train]
            for subject in range(X_train.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_train, subj_y_train = np.expand_dims(X_train[subject], axis=0), np.expand_dims(y_train[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)
                # TODO: Might be dangerous to balance here
                subj_X_train, subj_y_train = balance(rf_inputs, rf_outputs)
                train_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_train' + ext_mem_extension)
                save_to_svmlight(subj_X_train, subj_y_train, train_data_path)

            X_test, y_test = X[test], y[test]
            for subject in range(X_test.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_test, subj_y_test = np.expand_dims(X_test[subject], axis=0), np.expand_dims(y_test[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_test, subj_y_test, receptive_field_dimensions)
                test_data_path = os.path.join(fold_dir, 'fold_' + str(fold) + '_test' + ext_mem_extension)
                save_to_svmlight(rf_inputs, rf_outputs, test_data_path)

            fold += 1

    end = timeit.default_timer()
    print('Created and saved splits in: ', end - start)

def external_evaluate_patient_wise_kfold_cv(params, data_dir):
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

            dtrain = xgb.DMatrix(os.path.join(fold_dir, fold + '_train' + ext_mem_extension)
                + '#' + os.path.join(fold_dir, 'dtrain.cache'))
            dtest = xgb.DMatrix(os.path.join(fold_dir, fold + '_test' + ext_mem_extension)
                + '#' + os.path.join(fold_dir, 'dtest.cache'))

            trained_model = xgb.train(params, dtrain)

            # fold_name_pure = fold.split('.')[0]
            # model_extension = '.pkl'
            # model_path = os.path.join(iteration_dir, 'model_' + fold_name_pure + model_extension)
            # print('Saving model as : ', model_path)
            # joblib.dump(trained_model, model_path)

            print('Testing', fold, 'of', iteration)
            y_test = dtest.get_label()
            probas_= trained_model.predict(dtest)
            # Compute ROC curve, area under the curve, f1, and accuracy
            threshold = 0.5 # threshold choosen ot evaluate f1 and accuracy of model
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:])
            accuracies.append(accuracy_score(y_test, probas_[:] > threshold))
            f1_scores.append(f1_score(y_test, probas_[:] > threshold))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(tpr)
            fprs.append(fpr)

    return {
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'model': params,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_TPR': tprs,
        'test_FPR': fprs
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


def balance(X, y, verbose = False):
    # Prefer under_sampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, y)

    # Avoid over sampling because it overloads the RAM
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, y)

    if (verbose):
        print('Balancing Data.')
        print('Remaining data points after balancing: ', sorted(Counter(y_resampled).items()))

    return (X_resampled, y_resampled)
