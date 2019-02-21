import os, sys, shutil, traceback, timeit
sys.path.insert(0, '../')
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sampling_utils import get_undersample_selector_array
import receptiveField as rf
from scoring_utils import evaluate

def repeated_kfold_cv(Model_Generator, save_dir, save_function,
            input_data_array, output_data_array, clinical_input_array = None, mask_array = None, feature_scaling = True,
            receptive_field_dimensions = [1,1,1], n_repeats = 1, n_folds = 5, messaging = None):
    """
    Patient wise Repeated KFold Crossvalidation for a given model
    This function creates and evaluates k datafolds of n-iterations for crossvalidation

    Args:
        Model_Generator: initialises a given model
        save_dir: directory to use for saving the intermittent states
        save_function: function for saving the states --> save(results, trained_models, figures)
        clinX (optional): clinical input data to validate for all subjects in form of a list [subject, clinical_data]
        imgX: image input data to validate for all subjects in form of an np array [subject, x, y, z, c]
        y: dependent variables of data in a form of an np array [subject, x, y, z]
        mask_array: boolean array differentiating brain from brackground
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

    if len(imgX.shape) < 5:
        imgX = np.expand_dims(imgX, axis=5)

    model_params = Model_Generator.get_settings()
    Model_Generator.hello_world()

    # Initialising variables for evaluation
    used_clinical = False
    if clinX is not None:
        used_clinical = True
    used_brain_masking = False
    if mask_array is not None:
        used_brain_masking = True
    failed_folds = 0
    trained_models = []
    figures = []
    results = {
        'params': {
            'model_params': model_params,
            'rf': receptive_field_dimensions,
            'used_clinical': used_clinical,
            'masked_background': used_brain_masking,
            'scaled': feature_scaling,
            'settings_repeats': n_repeats,
            'settings_folds': n_folds,
            'settings_imgX_shape': imgX.shape,
            'settings_y_shape': y.shape,
            'failed_folds': failed_folds
        },
        'train_evals': [],
        'test_accuracy': [],
        'test_roc_auc': [],
        'test_f1': [],
        'test_jaccard': [],
        'test_TPR': [],
        'test_FPR': [],
        'test_thresholded_volume_deltas': [],
        'test_unthresholded_volume_deltas': [],
        'test_image_wise_error_ratios': [],
        'test_image_wise_jaccards': [],
        'test_image_wise_hausdorff': [],
        'test_image_wise_modified_hausdorff': [],
        'test_image_wise_dice': [],
        'evaluation_thresholds': []
    }

    print('Repeated kfold', n_repeats, n_folds)

    print('Input image data shape:', imgX.shape)
    n_x, n_y, n_z, n_c = imgX[0].shape
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

    # Standardise data (data - mean / std)
    if feature_scaling == True:
        imgX, clinX = standardise(imgX, clinX)

    # Start iteration of repeated_kfold_cv
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
            model = Model_Generator(fold_dir, fold, n_channels = n_c, n_channels_out = 1, rf = receptive_field_dimensions)

            # Create this fold
            try:
                print('Creating fold : ' + str(fold))
                create_fold(model, imgX, y, mask_array, receptive_field_dimensions, train, test, clinX = clinX)
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
                fold_result = evaluate_fold(model, n_test_subjects, n_x, n_y, n_z, mask_array, test)

                results['test_accuracy'].append(fold_result['accuracy'])
                results['test_f1'].append(fold_result['f1'])
                results['test_roc_auc'].append(fold_result['roc_auc'])
                results['test_TPR'].append(fold_result['tpr'])
                results['test_FPR'].append(fold_result['fpr'])
                results['test_jaccard'].append(fold_result['jaccard'])
                results['test_thresholded_volume_deltas'].append(fold_result['thresholded_volume_deltas'])
                results['test_unthresholded_volume_deltas'].append(fold_result['unthresholded_volume_deltas'])
                results['test_image_wise_error_ratios'].append(fold_result['image_wise_error_ratios'])
                results['test_image_wise_jaccards'].append(fold_result['image_wise_jaccards'])
                results['test_image_wise_hausdorff'].append(fold_result['image_wise_hausdorff'])
                results['test_image_wise_modified_hausdorff'].append(fold_result['image_wise_modified_hausdorff'])
                results['test_image_wise_dice'].append(fold_result['image_wise_dice'])
                results['train_evals'].append(fold_result['train_evals'])
                results['evaluation_thresholds'].append(fold_result['evaluation_threshold'])
                trained_models.append(fold_result['trained_model'])
                figures.append(fold_result['figure'])
                pass
            except Exception as e:
                results['failed_folds'] += 1
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

            # save current state of progression
            save_function(results, trained_models, figures)

            fold += 1
            # End of fold iteration

        try:
            shutil.rmtree(iteration_dir)
        except:
            print('No iteration to clear.')
        # End of iteration iteration

    end = timeit.default_timer()
    print('Created, saved and evaluated splits in: ', str(end - start))

    return (results, trained_models)

def create_fold(model, imgX, y, mask_array, receptive_field_dimensions, train, test, clinX = None):
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

    imgX_train, y_train, mask_train = imgX[train], y[train], mask_array[train]
    if clinX is not None:
        clinX_train = clinX[train]

    # Get balancing selector --> random subset respecting population wide distribution
    # Balancing chooses only data inside the brain (mask is applied through balancing)
    balancing_selector = get_undersample_selector_array(y_train, mask_train)

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


    X_test, y_test, mask_test = imgX[test], y[test], mask_array[test]
    if clinX is not None:
        clinX_test = clinX[test]

    model.initialise_test_data(np.sum(mask_test), input_size)

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

        subj_X_test, subj_y_test = all_inputs[mask_test[subject].reshape(-1)], rf_outputs[mask_test[subject].reshape(-1)]

        model.add_test_data(subj_X_test, subj_y_test)

def evaluate_fold(model, n_test_subjects, n_x, n_y, n_z, mask_array, test):
    """
    Patient wise Repeated KFold Crossvalidation
    This function evaluates a saved datafold
    Args:
        model
        n_test_subjects

    Returns: result dictionary
    """

    trained_model, evals_result = model.train()
    print('Model sucessfully trained.')
    probas_ = model.predict_test_data()
    y_test = model.get_test_labels()
    mask_test = mask_array[test]

    results = evaluate(probas_, y_test, mask_test, n_test_subjects, n_x, n_y, n_z)
    print('Model sucessfully tested.')
    results['trained_model'] = trained_model
    results['train_evals'] = evals_result

    return results

def standardise(imgX, clinX):
    original_shape = imgX.shape
    imgX = imgX.reshape(-1, imgX.shape[-1])
    scaler = StandardScaler(copy = False)
    rescaled_imgX = scaler.fit_transform(imgX).reshape(original_shape)
    if clinX is not None:
        rescaled_clinX = scaler.fit_transform(clinX)
    else:
        rescaled_clinX = clinX
    return rescaled_imgX, rescaled_clinX
