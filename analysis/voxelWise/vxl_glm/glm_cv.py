import sys, os, timeit, traceback
sys.path.insert(0, '../')

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, jaccard_similarity_score, roc_auc_score, precision_score, roc_curve, auc, accuracy_score
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score, cross_validate
from sampling_utils import balance, get_undersample_selector_array
import receptiveField as rf
from scoring_utils import evaluate


def glm_continuous_repeated_kfold_cv(imgX, y, receptive_field_dimensions, clinX = None, n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for glm
    This function creates and evaluates k datafolds of n-iterations for crossvalidation,

    Args:
        imgX: data to validate for all subjects in form of an np array [subject, x, y, z, c]
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
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
    jaccards = []
    thresholded_volume_deltas = []
    unthresholded_volume_deltas = []
    image_wise_error_ratios = []
    image_wise_jaccards = []
    trained_models = []
    failed_folds = 0

    if clinX is not None:
        print('Using clinical data.')
        if clinX.shape[0] != imgX.shape[0]:
            raise ValueError('Not the same number of clinical and imaging data points:', clinX.shape, imgX.shape)

    n_x, n_y, n_z, n_c = imgX[0].shape
    rf_x, rf_y, rf_z = receptive_field_dimensions
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c

    start = timeit.default_timer()
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration += 1
        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(imgX, y):
            print('Creating fold : ' + str(fold))
            # Create a new glm model for every fold
            glm_model = linear_model.LogisticRegression(verbose = 0, max_iter = 1000000000, n_jobs = -1)

            input_size = receptive_field_size
            if clinX is not None:
                input_size += clinX[0].size

            X_train, y_train = imgX[train], y[train]
            if clinX is not None:
                clinX_train = clinX[train]

            # Get balancing selector respecting population wide distribution
            balancing_selector = get_undersample_selector_array(y_train)

            # Initialising arrays that will contain all data
            all_subj_X_train = np.empty([np.sum(balancing_selector), input_size])
            all_subj_y_train = np.empty(np.sum(balancing_selector))
            all_subj_index = 0

            for subject in range(X_train.shape[0]):
                # reshape to rf expects data with n_subjects in first
                subj_X_train, subj_y_train = np.expand_dims(X_train[subject], axis=0), np.expand_dims(y_train[subject], axis=0)
                rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)

                if clinX is not None:
                    # Add clinical data to every voxel
                    # As discussed here: https://stackoverflow.com/questions/52132331/how-to-add-multiple-extra-columns-to-a-numpy-array/52132400#52132400
                    subj_mixed_inputs = np.zeros((rf_inputs.shape[0], input_size), dtype = np.float) # Initialising matrix of the right size
                    subj_mixed_inputs[:, : rf_inputs.shape[1]] = rf_inputs
                    subj_mixed_inputs[:, rf_inputs.shape[1] :]= clinX_train[subject]
                    all_inputs = subj_mixed_inputs
                else:
                    all_inputs = rf_inputs

                # Balance by using predefined balancing_selector
                subj_X_train, subj_y_train = all_inputs[balancing_selector[subject].reshape(-1)], rf_outputs[balancing_selector[subject].reshape(-1)]
                all_subj_X_train[all_subj_index : all_subj_index + subj_X_train.shape[0], :] = subj_X_train
                all_subj_y_train[all_subj_index : all_subj_index + subj_y_train.shape[0]] = subj_y_train
                all_subj_index += subj_X_train.shape[0]

            glm_model.fit(all_subj_X_train, all_subj_y_train)

            X_test, y_test = imgX[test], y[test]
            if clinX is not None:
                clinX_test = clinX[test]
            n_test_subjects = X_test.shape[0]
            test_rf_inputs, test_rf_outputs = rf.reshape_to_receptive_field(X_test, y_test, receptive_field_dimensions)
            if clinX is not None:
                # Add clinical data to every voxel
                subj_mixed_inputs = np.zeros((test_rf_inputs.shape[0], input_size), np.float) # Initialising matrix of the right size
                subj_mixed_inputs[:, : test_rf_inputs.shape[1]] = test_rf_inputs
                subj_mixed_inputs[:, test_rf_inputs.shape[1] :]= clinX_test
                all_test_inputs = subj_mixed_inputs
            else:
                all_test_inputs = test_rf_inputs

            # Evaluate this fold
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration))

            try:
                fold_result = glm_evaluate_fold_cv(all_test_inputs, test_rf_outputs, n_test_subjects, glm_model)
                accuracies.append(fold_result['accuracy'])
                f1_scores.append(fold_result['f1'])
                aucs.append(fold_result['roc_auc'])
                tprs.append(fold_result['tpr'])
                fprs.append(fold_result['fpr'])
                jaccards.append(fold_result['jaccard'])
                thresholded_volume_deltas.append(fold_result['thresholded_volume_deltas'])
                unthresholded_volume_deltas.append(fold_result['unthresholded_volume_deltas'])
                image_wise_error_ratios.append(fold_result['image_wise_error_ratios'])
                image_wise_jaccards.append(fold_result['image_wise_jaccards'])
                trained_models.append(fold_result['trained_model'])
                pass
            except Exception as e:
                failed_folds += 1
                print('Evaluation of fold failed.')
                print(e)
                tb = traceback.format_exc()
                print(tb)
                if (messaging):
                    title = 'Minor error upon rf_hyperopt at ' + str(receptive_field_dimensions)
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            fold += 1
            # End of fold iteration
        # End of iteration iteration

    end = timeit.default_timer()
    print('Created, saved and evaluated splits in: ', end - start)


    used_clinical = False
    if clinX is not None:
        used_clinical = True


    return ({
        'rf': receptive_field_dimensions,
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'failed_folds': failed_folds,
        'model_params': 'glm',
        'used_clinical': used_clinical,
        'test_accuracy': accuracies,
        'test_roc_auc': aucs,
        'test_f1': f1_scores,
        'test_jaccard': jaccards,
        'test_TPR': tprs,
        'test_FPR': fprs,
        'test_thresholded_volume_deltas': thresholded_volume_deltas,
        'test_unthresholded_volume_deltas': unthresholded_volume_deltas,
        'test_image_wise_error_ratios': image_wise_error_ratios,
        'test_image_wise_jaccards': image_wise_jaccards
    },
        trained_models
    )


def glm_evaluate_fold_cv(X_test, y_test, n_test_subjects, trained_model):
    """
    Patient wise Repeated KFold Crossvalidation for glm
    This function evaluates a saved datafold

    Args: X_test, y_test, n_test_subjects, trained_model

    Returns: result dictionary
    """
    probas_= trained_model.predict_proba(X_test)
    probas_ = probas_[:,1]

    results = evaluate(probas_, y_test, n_test_subjects)
    results['trained_model'] = trained_model

    return results
