import os, timeit, traceback
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sampling_utils import balance, get_undersample_selector_array
import receptiveField as rf


def glm_continuous_repeated_kfold_cv(X, y, receptive_field_dimensions, n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for glm
    This function creates and evaluates k datafolds of n-iterations for crossvalidation,

    Args:
        X: data to validate for all subjects in form of an np array [subject, x, y, z, c]
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
    print('CONTINOUS REPEATED KFOLD CV FOR GLM')

    # Initialising variables for evaluation
    tprs = []
    fprs = []
    aucs = []
    accuracies = []
    f1_scores = []
    failed_folds = 0

    start = timeit.default_timer()
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(X, y):
            print('Creating fold : ' + str(fold))
            glm_model = linear_model.LogisticRegression(verbose = 1, max_iter = 1000000000, n_jobs = -1)

            X_train, y_train = X[train], y[train]

            # Get balancing selector respecting population wide distribution
            balancing_selector = get_undersample_selector_array(y_train)

            for subject in range(X_train.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_train, subj_y_train = np.expand_dims(X_train[subject], axis=0), np.expand_dims(y_train[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

                # Balance by using predefined balancing_selector
                subj_X_train, subj_y_train = rf_inputs[balancing_selector[subject].reshape(-1)], rf_outputs[balancing_selector[subject].reshape(-1)]

            glm_model.fit(X_train_input, y_train_input)

            X_test, y_test = X[test], y[test]
            for subject in range(X_test.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_test, subj_y_test = np.expand_dims(X_test[subject], axis=0), np.expand_dims(y_test[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_test, subj_y_test, receptive_field_dimensions)

            # Evaluate this fold
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration))

            try:
                fold_result = glm_evaluate_fold_cv(fold data)
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
                    title = 'Minor error upon rf_hyperopt at ' + str(receptive_field_dimensions)
                    tb = traceback.format_exc()
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            fold += 1
            # End of fold iteration
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


def glm_evaluate_fold_cv():
    """
    Patient wise Repeated KFold Crossvalidation for glm
    This function evaluates a saved datafold

    Args:

    Returns: result dictionary
    """
    n_estimators = params['n_estimators']

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
