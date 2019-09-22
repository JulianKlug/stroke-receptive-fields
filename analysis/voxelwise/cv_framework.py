import os, sys, shutil, traceback, timeit
sys.path.insert(0, '../')
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sampling_utils import get_undersample_selector_array
import voxelwise.receptiveField as rf
from voxelwise.scoring_utils import evaluate
from utils import gaussian_smoothing, rescale_outliers, standardise
from voxelwise.penumbra_evaluation import penumbra_match
from voxelwise.channel_normalisation import normalise_channel_by_contralateral

def repeated_kfold_cv(Model_Generator, save_dir, save_function,
            input_data_array, output_data_array, clinical_input_array = None, mask_array = None, id_array = None,
            feature_scaling = True, pre_smoothing = False, channels_to_normalise = False, undef_normalisation = True,
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
        mask_array: boolean array differentiating brain from background
        id_array: array with subj ids
        feature_scalinng: boolean if data should be normalised
        pre_smoothing: boolean if gaussian smoothing should be applied on all images slices of z
        channel_normalisation: False or array of channels to normalise
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
            'smoothed_beforehand': pre_smoothing,
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
        'test_roc_thresholds': [],
        'test_positive_predictive_value': [],
        'test_thresholded_predicted_volume_vox': [],
        'test_thresholded_volume_deltas': [],
        'test_unthresholded_volume_deltas': [],
        'test_image_wise_error_ratios': [],
        'test_image_wise_jaccards': [],
        'test_image_wise_hausdorff': [],
        'test_image_wise_modified_hausdorff': [],
        'test_image_wise_dice': [],
        'test_image_wise_roc_auc': [],
        'test_image_wise_fpr': [],
        'test_image_wise_tpr': [],
        'test_image_wise_roc_thresholds': [],
        'evaluation_thresholds': [],
        'optimal_thresholds_on_test_data': [],
        'test_penumbra_metrics': {
            'predicted_in_penumbra_ratio': []
        }
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

    # rescale outliers
    imgX = rescale_outliers(imgX, MASKS = mask_array)

    # Standardise data (data - mean / std)
    if feature_scaling:
        imgX, clinX = standardise(imgX, clinX)

    # Smooth data with a gaussian Kernel before using it for training/testing
    if pre_smoothing:
        imgX = gaussian_smoothing(imgX)

    # Normalise channels by contralateral side before using them for training / testing
    if channels_to_normalise:
        for c in channels_to_normalise:
            image_normalised_channel, flat_normalised_channel = normalise_channel_by_contralateral(imgX[mask_array], mask_array, c)
            imgX[..., c] = image_normalised_channel

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
                create_fold(model, imgX, y, mask_array, receptive_field_dimensions, train, test, clinX = clinX, undef_normalisation = undef_normalisation)
            except Exception as e:
                tb = traceback.format_exc()
                print('Creation of fold failed.')
                print(e)
                print(tb)
                if (messaging):
                    title = 'Minor error upon fold creation rf_hyperopt at ' + str(receptive_field_dimensions) + ' in ' + str(save_dir)
                    body = 'RF ' + str(receptive_field_dimensions) + '\n' + 'fold ' + str(fold) + '\n' +'iteration ' + str(iteration) + '\n' + 'Error ' + str(e) + '\n' + str(tb)
                    messaging.send_message(title, body)

            # Evaluate this fold
            print('Evaluating fold ' + str(fold) + ' of ' + str(n_folds - 1) + ' of iteration' + str(iteration) + ' in', str(fold_dir))
            try:
                fold_result = evaluate_fold(model, n_test_subjects, n_x, n_y, n_z, imgX, mask_array, id_array, test)

                results['test_accuracy'].append(fold_result['accuracy'])
                results['test_f1'].append(fold_result['f1'])
                results['test_roc_auc'].append(fold_result['roc_auc'])
                results['test_TPR'].append(fold_result['tpr'])
                results['test_FPR'].append(fold_result['fpr'])
                results['test_roc_thresholds'].append(fold_result['roc_thresholds'])
                results['test_jaccard'].append(fold_result['jaccard'])
                results['test_positive_predictive_value'].append(fold_result['positive_predictive_value'])
                results['test_thresholded_predicted_volume_vox'].append(fold_result['thresholded_predicted_volume_vox'])
                results['test_thresholded_volume_deltas'].append(fold_result['thresholded_volume_deltas'])
                results['test_unthresholded_volume_deltas'].append(fold_result['unthresholded_volume_deltas'])
                results['test_image_wise_error_ratios'].append(fold_result['image_wise_error_ratios'])
                results['test_image_wise_jaccards'].append(fold_result['image_wise_jaccards'])
                results['test_image_wise_hausdorff'].append(fold_result['image_wise_hausdorff'])
                results['test_image_wise_modified_hausdorff'].append(fold_result['image_wise_modified_hausdorff'])
                results['test_image_wise_dice'].append(fold_result['image_wise_dice'])
                results['test_image_wise_roc_auc'].append(fold_result['image_wise_roc_auc'])
                results['test_image_wise_fpr'].append(fold_result['image_wise_fpr'])
                results['test_image_wise_tpr'].append(fold_result['image_wise_tpr'])
                results['test_image_wise_roc_thresholds'].append(fold_result['image_wise_roc_thresholds'])
                results['train_evals'].append(fold_result['train_evals'])
                if not (fold_result['penumbra_metrics'] is None):
                    results['test_penumbra_metrics']['predicted_in_penumbra_ratio']\
                        .append(fold_result['penumbra_metrics']['predicted_in_penumbra_ratio'])
                results['evaluation_thresholds'].append(fold_result['evaluation_threshold'])
                results['optimal_thresholds_on_test_data'].append(fold_result['optimal_threshold_on_test_data'])
                trained_models.append(fold_result['trained_model'])
                figures.append(fold_result['figure'])
                pass
            except Exception as e:
                results['params']['failed_folds'] += 1
                print('Evaluation of fold failed.')
                tb = traceback.format_exc()
                print(e)
                print(tb)

                if (messaging):
                    title = 'Minor error upon fold evaluation rf_hyperopt at ' + str(receptive_field_dimensions) + ' in ' + str(save_dir)
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

def create_fold(model, imgX, y, mask_array, receptive_field_dimensions, train, test, clinX = None, undef_normalisation = True):
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
    n_train, n_test = len(train), len(test)
    window_d_x, window_d_y, window_d_z = 2 * np.array(receptive_field_dimensions) + 1
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c
    # defines the size of X
    input_size = receptive_field_size

    # If a normalisation term for undefined brain areas should be added, the input size will be
    # receptive_field_size + 1 for a normalisation term
    if np.max(receptive_field_dimensions) > 0 and undef_normalisation:
        input_size += 1

    if clinX is not None:
        input_size += clinX[0].size

    imgX_train, y_train, mask_train = imgX[train], y[train], mask_array[train]
    if clinX is not None:
        clinX_train = clinX[train]

    # Get balancing selector --> random subset respecting population wide distribution
    # Balancing chooses only data inside the brain (mask is applied through balancing)
    balancing_selector = get_undersample_selector_array(y_train, mask_train)

    model.initialise_train_data(balancing_selector, input_size, n_train, (n_x, n_y, n_z))

    for subject in range(imgX_train.shape[0]):
        subjects_per_batch = 1

        # reshape to rf expects data with n_subjects in first dimension (here 1)
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


        # Add a normalization term to account for the number of voxels outside the defined brain in a receptive field
        if np.max(receptive_field_dimensions) > 0 and undef_normalisation:
            undef_normalisation_terms = rf.cardinal_undef_in_receptive_field(
                np.expand_dims(mask_array[subject], axis=0), receptive_field_dimensions)
            all_inputs = np.concatenate((all_inputs, undef_normalisation_terms.reshape(-1, 1)), axis=1)

        # Balance by using predefined balancing_selector
        selected_for_training = balancing_selector[subject].reshape(-1)
        subj_X_train, subj_y_train = all_inputs[selected_for_training], rf_outputs[selected_for_training]

        model.add_train_data(subj_X_train, subj_y_train, selected_for_training, subjects_per_batch)


    X_test, y_test, mask_test = imgX[test], y[test], mask_array[test]
    if clinX is not None:
        clinX_test = clinX[test]


    model.initialise_test_data(np.sum(mask_test), input_size, n_test, (n_x, n_y, n_z))

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

        # Add a normalization term to account for the number of voxels outside the defined brain in a receptive field
        if np.max(receptive_field_dimensions) > 0 and undef_normalisation:
            undef_normalisation_terms = rf.cardinal_undef_in_receptive_field(
                np.expand_dims(mask_array[subject], axis=0), receptive_field_dimensions)
            all_inputs = np.concatenate((all_inputs, undef_normalisation_terms.reshape(-1, 1)), axis=1)

        selected_for_testing = mask_test[subject].reshape(-1)
        subj_X_test, subj_y_test = all_inputs[selected_for_testing], rf_outputs[selected_for_testing]

        model.add_test_data(subj_X_test, subj_y_test, selected_for_testing, subjects_per_batch)

def evaluate_fold(model, n_test_subjects, n_x, n_y, n_z, imgX, mask_array, id_array, test):
    """
    Patient wise Repeated KFold Crossvalidation
    This function evaluates a saved datafold
    Args:
        model
        n_test_subjects

    Returns: result dictionary
    """

    trained_model, model_threshold, evals_result = model.train()
    print('Model successfully trained. Threshold at:', str(model_threshold))
    probas_ = model.predict_test_data()
    y_test = model.get_test_labels()
    mask_test = mask_array[test]
    imgX_test = imgX[test][mask_test]
    if id_array is not None: ids_test = id_array[test]
    else: ids_test = None

    results = evaluate(probas_, y_test, mask_test, ids_test, n_test_subjects, n_x, n_y, n_z, model_threshold)
    print('Model successfully tested.')
    results['trained_model'] = trained_model
    results['train_evals'] = evals_result
    results['trained_threshold'] = model_threshold

    # evaluation of relation with penumbra
    if imgX_test.shape[-1] == 4: # evaluation can only be done if all channels are there (Tmax must be present)
        results['penumbra_metrics'] = {
            'predicted_in_penumbra_ratio': penumbra_match(probas_, imgX_test)
        }
    else:
        results['penumbra_metrics'] = None


    return results


