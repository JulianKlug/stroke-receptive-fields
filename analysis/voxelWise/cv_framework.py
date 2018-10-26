import os, sys, shutil, traceback, timeit
sys.path.insert(0, '../')
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sampling_utils import get_undersample_selector_array
import receptiveField as rf
from scoring_utils import evaluate

def repeated_kfold_cv(Model_Generator, save_dir,
            input_data_array, output_data_array, clinical_input_array = None,
            receptive_field_dimensions = [1,1,1], n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for a given model
    This function creates and evaluates k datafolds of n-iterations for crossvalidation

    Args:
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
    imgX = input_data_array
    y = output_data_array
    clinX = clinical_input_array

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
    train_evals = []
    failed_folds = 0

    model_params = Model_Generator.get_settings()
    Model_Generator.hello_world()

    print('Repeated kfold', n_repeats, n_folds)

    if len(imgX.shape) < 5:
        imgX = np.expand_dims(imgX, axis=5)

    print('Input image data shape:', imgX.shape)
    if clinX is not None:
        print('Using clinical data.')
        if clinX.shape[0] != imgX.shape[0]:
            raise ValueError('Not the same number of clinical and imaging data points:', clinX.shape, imgX.shape)
    else:
        print('Not using clinical data')

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
        print('This directory already exists: ', save_dir)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Save Dir for Model already exists. Choose another model name or delete current model')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    start = timeit.default_timer()

    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration += 1
        iteration_dir = os.path.join(save_dir, 'iteration_' + str(iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)

        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = KFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train, test in kf.split(imgX, y):
            n_test_subjects = test.size
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            # Create a new model for every fold
            model = Model_Generator(fold_dir, fold)

            # Create this fold
            try:
                print('Creating fold : ' + str(fold))
                create_fold(model, imgX, y, receptive_field_dimensions, train, test, clinX = clinX)
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
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration) + ' in', str(fold_dir))
            try:
                fold_result = evaluate_fold(model, n_test_subjects)

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
                train_evals.append(fold_result['train_evals'])
                trained_models.append(fold_result['trained_model'])
                pass
            except Exception as e:
                failed_folds += 1
                print('Evaluation of fold failed.')
                tb = traceback.format_exc()
                print(e)
                print(tb)

                if (messaging):
                    title = 'Minor error upon fold evaluation rf_hyperopt at ' + str(receptive_field_dimensions)
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
    print('Created, saved and evaluated splits in: ', str(end - start))

    used_clinical = False
    if clinX is not None:
        used_clinical = True

    return ({
        'settings_repeats': n_repeats,
        'settings_folds': n_folds,
        'failed_folds': failed_folds,
        'model_params': model_params,
        'rf': receptive_field_dimensions,
        'used_clinical': used_clinical,
        'train_evals': train_evals,
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


def create_fold(model, imgX, y, receptive_field_dimensions, train, test, clinX = None):
    """
    Create a fold given the data and the test / train distribution
    External Memory: saves the folds as libsvm files

    Args:
        model
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
        imgX: image input data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        receptive_field_dimensions : in the form of a list as  [rf_x, rf_y, rf_z]
        train: boolean array selecting for Training
        test: boolean array selecting for testing

    Returns: undefined
    """

    n_x, n_y, n_z, n_c = imgX[0].shape
    window_d_x, window_d_y, window_d_z  = 2 * np.array(receptive_field_dimensions) + 1
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c
    # defines the size of X
    input_size = receptive_field_size

    if clinX is not None:
        input_size += clinX[0].size

    imgX_train, y_train = imgX[train], y[train]
    if clinX is not None:
        clinX_train = clinX[train]

    # Get balancing selector --> random subset respecting population wide distribution
    balancing_selector = get_undersample_selector_array(y_train)

    model.initialise_train_data(balancing_selector, input_size)

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

        model.add_train_data(subj_X_train, subj_y_train)


    X_test, y_test = imgX[test], y[test]
    if clinX is not None:
        clinX_test = clinX[test]

    n_vox_per_subj = n_x * n_y * n_z
    model.initialise_test_data(n_vox_per_subj * X_test.shape[0], input_size)

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

        model.add_test_data(all_inputs, rf_outputs)

def evaluate_fold(model, n_test_subjects):
    """
    Patient wise Repeated KFold Crossvalidation
    This function evaluates a saved datafold
    Args:
        model
        n_test_subjects

    Returns: result dictionary
    """

    trained_model, evals_result = model.train()
    probas_ = model.predict_test_data()
    y_test = model.get_test_labels()

    results = evaluate(probas_, y_test, n_test_subjects)
    results['trained_model'] = trained_model
    results['train_evals'] = evals_result

    return results
