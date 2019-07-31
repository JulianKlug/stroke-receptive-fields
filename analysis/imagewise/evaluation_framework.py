import sys, os, json
sys.path.insert(0, '../')
import torch
import numpy as np
from scoring_utils import evaluate_imagewise
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def train_test_evaluation(experiment_prefix, model, input_data, gt_data, mask_data, subject_ids, save_dir, config):
    experiment_name = experiment_prefix + '_' + model.model_name
    print('Running:', experiment_name)
    model.hello_world()

    # initialise folder to save to
    save_folder = os.path.join(save_dir, experiment_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print('Overwriting saved data for old run of:', experiment_name)

    # remove outliers (some subjects seem to be scaled x10)
    input_data = rescale_outliers(input_data, MASKS = mask_data)

    # data normalisation
    if model.get_settings()['input_pre_normalisation']:
        input_data = standardise(input_data)

    # todo possibly add smoothing

    # train / test split
    x_train, x_test, y_train, y_test, mask_train, mask_test, ids_train, ids_test = model_selection.train_test_split(input_data, gt_data, mask_data, subject_ids, test_size=0.3, random_state=42)

    log_dir = os.path.join(save_folder, 'logs')
    print('Logging to:', log_dir)
    model, evals = model.train(x_train, y_train, mask_train, log_dir)
    model_threshold = model.get_threshold()
    test_proba_predictions = model.predict(x_test, mask_test)
    results, figure = evaluate_imagewise(test_proba_predictions, y_test, mask_test, ids_test, model_threshold)
    print(results_report(results))

    params = {
        'model_name': model.model_name,
        'model_settings': model.get_settings(),
        'evaluation_method': "train/test split",
        'used_clinical': False,
        'masked_background': model.get_settings()['used_brain_masking'],
        'evaluation_config': config
    }

    save_results(save_folder, experiment_name, model, results, params, figure)


def save_results(save_dir, experiment_name, trained_model, results, params, figure):
    torch.save(results, os.path.join(save_dir, 'scores_' + experiment_name + '.npy'))

    trained_model.save(os.path.join(save_dir, 'trained_model_' + experiment_name + '.h5'))
    # torch.save(trained_model, os.path.join(save_dir, 'trained_models_' + experiment_name + '.npy'))

    with open(os.path.join(save_dir, 'params_' + experiment_name + '.json'), 'w') as fp:
        json.dump(params, fp, sort_keys=False, indent=4)

    # save plots
    plt.ioff()
    plt.switch_backend('agg')
    figure_path = os.path.join(save_dir, experiment_name + '_test_predictions')
    figure.savefig(figure_path, dpi='figure')
    plt.close(figure)

def rescale_outliers(imgX, MASKS):
    '''
    Rescale outliers as some images from RAPID seem to be scaled x10
    Outliers are detected if their median exceeds 5 times the global median and are rescaled by dividing through 10
    :param imgX: image data (n, x, y, z, c)
    :return: rescaled_imgX
    '''

    for i in range(imgX.shape[0]):
        for channel in range(imgX.shape[-1]):
            median_channel = np.median(imgX[..., channel][MASKS])
            if np.median(imgX[i, ..., 0][MASKS[i]]) > 5 * median_channel:
                imgX[i, ..., 0] = imgX[i, ..., channel] / 10

    return imgX

def standardise(imgX):
    original_shape = imgX.shape
    imgX = imgX.reshape(-1, imgX.shape[-1])
    scaler = StandardScaler(copy = False)
    rescaled_imgX = scaler.fit_transform(imgX).reshape(original_shape)
    return rescaled_imgX

def results_report(results):
    report = f'Total ROC AUC: {results["total_roc_auc"]} \n' \
             f'Dice: {np.median(results["dice_coefficient"])} \n' \
             f'Hausdorff: {np.median(results["hausdorff_distance"])}'
    return report