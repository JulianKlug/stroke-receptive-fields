import sys, os, shutil, torch
sys.path.insert(0, '../')
import numpy as np

from utils import gaussian_smoothing, rescale_outliers, standardise
from sampling_utils import get_undersample_selector_array
import voxelwise.receptiveField as rf
from voxelwise.scoring_utils import evaluate
from voxelwise.penumbra_evaluation import penumbra_match
import matplotlib.pyplot as plt
from voxelwise.figures.train_test_evaluation import wrapper_plot_train_evaluation
from voxelwise.figures.plot_ROC import plot_roc
from email_notification import NotificationSystem

notification_system = NotificationSystem()

def evaluate_subgroup(Model_Generator, data_set, selected_indexes, save_dir,
                                  save_function, receptive_field_dimensions, feature_scaling, pre_smoothing):
    clinX, imgX, y, mri_inputs, mri_lesion_GT, mask_array, ids, param = data_set

    if len(imgX.shape) < 5:
        imgX = np.expand_dims(imgX, axis=5)

    print('Input image data shape:', imgX.shape)
    n_x, n_y, n_z, n_c = imgX[0].shape
    n_test_subjects = len(selected_indexes)

    model_params = Model_Generator.get_settings()
    Model_Generator.hello_world()
    model = Model_Generator('', '', n_channels=n_c, n_channels_out=1, rf=receptive_field_dimensions)

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
            'settings_repeats': 1,
            'settings_folds': 2,
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

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
        print('This directory already exists: ', save_dir)
        validation = input('Type `yes` if you wish to delete your previous data:\t')
        if (validation != 'yes'):
            raise ValueError('Save Dir for Model already exists. Choose another model name or delete current model')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # rescale outliers
    imgX = rescale_outliers(imgX, MASKS=mask_array)

    # Standardise data (data - mean / std)
    if feature_scaling:
        imgX, clinX = standardise(imgX, clinX)

    # Smooth data with a gaussian Kernel before using it for training/testing
    if pre_smoothing:
        imgX = gaussian_smoothing(imgX)

    imgX_test, y_test, mask_test = imgX[selected_indexes], y[selected_indexes], mask_array[selected_indexes]
    imgX_train, y_train, mask_train = np.delete(imgX, selected_indexes, axis=0),  np.delete(y, selected_indexes, axis=0), np.delete(mask_array, selected_indexes, axis=0)

    if ids is not None: ids_test = ids[selected_indexes]
    else: ids_test = None

    n_train, n_test = imgX_train.shape[0], imgX_test.shape[0]
    window_d_x, window_d_y, window_d_z = 2 * np.array(receptive_field_dimensions) + 1
    receptive_field_size = window_d_x * window_d_y * window_d_z * n_c
    # defines the size of X
    input_size = receptive_field_size

    balancing_selector = get_undersample_selector_array(y_train, mask_train)

    # prepare training data
    model.initialise_train_data(balancing_selector, input_size, n_train, (n_x, n_y, n_z))
    for subject in range(imgX_train.shape[0]):
        subjects_per_batch = 1

        # reshape to rf expects data with n_subjects in first dimension (here 1)
        subj_X_train, subj_y_train = np.expand_dims(imgX_train[subject], axis=0), np.expand_dims(y_train[subject],
                                                                                                 axis=0)
        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_train, subj_y_train, receptive_field_dimensions)
        all_inputs = rf_inputs

        # Balance by using predefined balancing_selector
        selected_for_training = balancing_selector[subject].reshape(-1)
        subj_X_train, subj_y_train = all_inputs[selected_for_training], rf_outputs[selected_for_training]

        model.add_train_data(subj_X_train, subj_y_train, selected_for_training, subjects_per_batch)

    # prepare test data
    model.initialise_test_data(np.sum(mask_test), input_size, n_test, (n_x, n_y, n_z))
    for subject in range(imgX_test.shape[0]):
        # reshape to rf expects data with n_subjects in first
        subj_X_test, subj_y_test = np.expand_dims(imgX_test[subject], axis=0), np.expand_dims(y_test[subject], axis=0)

        rf_inputs, rf_outputs = rf.reshape_to_receptive_field(subj_X_test, subj_y_test, receptive_field_dimensions)
        all_inputs = rf_inputs

        selected_for_testing = mask_test[subject].reshape(-1)
        subj_X_test, subj_y_test = all_inputs[selected_for_testing], rf_outputs[selected_for_testing]

        model.add_test_data(subj_X_test, subj_y_test, selected_for_testing, subjects_per_batch)

    #     train the model
    trained_model, model_threshold, evals_result = model.train()
    probas_ = model.predict_test_data()
    y_test = model.get_test_labels()
    imgX_test = imgX_test[mask_test]

    fold_result = evaluate(probas_, y_test, mask_test, ids_test, n_test_subjects, n_x, n_y, n_z, model_threshold)
    print('Model successfully tested.')
    fold_result['trained_model'] = trained_model
    fold_result['train_evals'] = evals_result
    fold_result['trained_threshold'] = model_threshold

    # evaluation of relation with penumbra
    if imgX_test.shape[-1] == 4:  # evaluation can only be done if all channels are there (Tmax must be present)
        fold_result['penumbra_metrics'] = {
            'predicted_in_penumbra_ratio': penumbra_match(probas_, imgX_test)
        }
    else:
        fold_result['penumbra_metrics'] = None

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
        results['test_penumbra_metrics']['predicted_in_penumbra_ratio'] \
            .append(fold_result['penumbra_metrics']['predicted_in_penumbra_ratio'])
    results['evaluation_thresholds'].append(fold_result['evaluation_threshold'])
    results['optimal_thresholds_on_test_data'].append(fold_result['optimal_threshold_on_test_data'])
    trained_models.append(fold_result['trained_model'])
    figures.append(fold_result['figure'])

    save_function(results, trained_models, figures)

    return results

def subgroup_evaluation_launcher(model_name, Model_Generator, data_set, selected_indexes, output_dir,
                                       receptive_field_dimensions, feature_scaling, pre_smoothing):
    def saveGenerator(output_dir, model_name):
        def save(results, trained_models, figures):
            # save the results and the params objects
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            visual_dir = os.path.join(output_dir, 'visual_check')
            if not os.path.exists(visual_dir):
                os.makedirs(visual_dir)

            torch.save(results, os.path.join(output_dir, 'scores_' + model_name + '.npy'))
            torch.save(results['params'], os.path.join(output_dir, 'params_' + model_name + '.npy'))
            torch.save(trained_models, os.path.join(output_dir, 'trained_models_' + model_name + '.npy'))
            wrapper_plot_train_evaluation(os.path.join(output_dir, 'scores_' + model_name + '.npy'), save_plot = True)
            plot_roc(results['test_TPR'], results['test_FPR'], output_dir, model_name, save_plot = True)

            plt.ioff()
            plt.switch_backend('agg')
            for i, figure in enumerate(figures):
                figure_path = os.path.join(visual_dir, model_name + '_test_predictions_fold_' + str(i))
                figure.savefig(figure_path, dpi='figure')
                plt.close(figure)

        return save
    save_function = saveGenerator(output_dir, model_name)

    results = evaluate_subgroup(Model_Generator, data_set, selected_indexes, output_dir,
                                  save_function, receptive_field_dimensions, feature_scaling, pre_smoothing)

    accuracy = np.median(results['test_accuracy'])
    roc_auc = np.median(results['test_roc_auc'])
    f1 = np.median(results['test_f1'])
    positive_predictive_value = np.median(results['test_positive_predictive_value'])
    dice = np.median([item for sublist in results['test_image_wise_dice'] for item in sublist])
    hausdorff_distance = np.median([item for sublist in results['test_image_wise_hausdorff'] for item in sublist])
    params = results['params']
    if not None in results['test_penumbra_metrics']['predicted_in_penumbra_ratio']:
        predicted_in_penumbra_ratio = np.median(results['test_penumbra_metrics']['predicted_in_penumbra_ratio'])
    else:
        predicted_in_penumbra_ratio = np.NaN

    print('Results for', model_name)
    print('Voxel-wise accuracy: ', accuracy)
    print('ROC AUC score: ', roc_auc)
    print('Dice score: ', dice)
    print('Classic Hausdorff', hausdorff_distance)
    print('Predicted_in_penumbra_ratio: ', predicted_in_penumbra_ratio)
    print('Positive predictive value:', positive_predictive_value)

    title = model_name + ' finished Cross-Validation'
    body = 'accuracy ' + str(accuracy) + '\n' + 'ROC AUC ' + str(roc_auc) + '\n' \
           + 'Dice ' + str(dice) + '\n' + 'Classic Hausdorff ' + str(hausdorff_distance) + '\n' \
           + 'Positive predictive value' + str(positive_predictive_value) + '\n' \
           + 'Predicted in penumbra ratio ' + str(predicted_in_penumbra_ratio) + '\n' \
           + 'RF ' + str(receptive_field_dimensions) + '\n' \
           + str(params)
    notification_system.send_message(title, body)

