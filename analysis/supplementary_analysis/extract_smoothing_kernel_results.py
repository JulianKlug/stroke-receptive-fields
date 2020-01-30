import os, torch, sys
import pandas as pd
import numpy as np


columns = ['model','rf', 'kernel_width' 'test_roc_auc', 'test_image_wise_dice', 'test_image_wise_hausdorff',
'test_accuracy', 'test_f1', 'test_jaccard', 'test_thresholded_volume_deltas', 'test_unthresholded_volume_deltas', 'test_image_wise_error_ratios', 'evaluation_thresholds']


def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]

def extract_results(main_dir, output_dir=None):
    if output_dir is None:
        output_dir = main_dir

    modalites = os.listdir(main_dir)

    for modality in modalites[0:]:
        modality_dir = os.path.join(main_dir, modality)
        if os.path.isdir(modality_dir):
            rf_evaluations = [o for o in os.listdir(modality_dir)
                                if os.path.isdir(os.path.join(modality_dir,o))]

            for rf_eval in rf_evaluations:
                print('Reading', rf_eval)
                rf_eval_dir = os.path.join(modality_dir, rf_eval)

                result_files = [i for i in os.listdir(rf_eval_dir)
                                    if os.path.isfile(os.path.join(rf_eval_dir, i))
                                        and i.startswith('scores_') and i.endswith('.npy')]
                results = torch.load(os.path.join(rf_eval_dir, result_files[0]))

                params_files = [i for i in os.listdir(rf_eval_dir)
                                    if os.path.isfile(os.path.join(rf_eval_dir, i))
                                        and i.startswith('params_') and i.endswith('.npy')]
                params = torch.load(os.path.join(rf_eval_dir, params_files[0]))

                model_name = '_'.join(rf_eval.split('_')[:-1])
                if 'params' in results:
                    params = results['params']
                try:
                    rf = int(np.mean(params['rf']))
                except (KeyError, TypeError):
                    rf = int(result_files[0].split('_')[-1].split('.')[0])

                kernel_width = int(result_files[0].split('_')[-3]).split('.')[0].split('k')[-1]

                mean_list = [[model_name, rf] + [np.mean(flatten(results[i])) if i in results else np.nan for i in columns[3:]]]
                std_list = [[model_name, rf] + [np.std(flatten(results[i])) if i in results else np.nan for i in columns[3:]]]
                median_list = [
                    [model_name, rf] + [np.median(flatten(results[i])) if i in results else np.nan for i in columns[3:]]]

                n_runs = len(results[columns[2]])
                current_all_list = np.concatenate((
                    np.repeat(model_name, n_runs).reshape(1, n_runs),
                    np.repeat(rf, n_runs).reshape(1, n_runs),
                    np.repeat(kernel_width, n_runs).reshape(1, n_runs),
                    [np.squeeze(np.array(results[i])) if i in results else np.repeat(np.nan, n_runs) for i in columns[3:]]
                    ))


                # make sure all result lists have the same size by filling up with nan
                max_runs = 50
                all_list = np.empty((len(columns), max_runs), dtype=object)
                all_list[:] = np.nan
                all_list[:, : n_runs] = current_all_list

                # coerce all variables to np arrays and floats
                for selector in range(1, all_list.shape[0]):
                    if type(all_list[selector][0]) == list or isinstance(all_list[selector][0], np.ndarray):
                        all_list[selector] = np.array([np.array(subL).astype(float) for subL in all_list[selector]])
                    else :
                        all_list[selector] = all_list[selector].astype(float)

                try:
                    all_results_array
                except NameError:
                    all_results_array = np.expand_dims(np.array(all_list), axis=0)
                else :
                    all_results_array = np.concatenate((all_results_array,
                                    np.expand_dims(np.array(all_list), axis=0)),
                                    axis = 0)
                try:
                    std_results_array
                except NameError:
                    std_results_array = np.array(std_list)
                else :
                    std_results_array = np.concatenate((std_results_array,
                                    np.array(std_list)))

                try:
                    mean_results_array
                except NameError:
                    mean_results_array = np.array(mean_list)
                else :
                    mean_results_array = np.concatenate((mean_results_array,
                                    np.array(mean_list)))

                try:
                    median_results_array
                except NameError:
                    median_results_array = np.array(median_list)
                else :
                    median_results_array = np.concatenate((median_results_array,
                                    np.array(median_list)))

    mean_results_df = pd.DataFrame(mean_results_array, columns = columns)
    std_results_df = pd.DataFrame(std_results_array, columns = columns)
    median_results_df = pd.DataFrame(median_results_array, columns = columns)



    with pd.ExcelWriter(os.path.join(output_dir, 'smoothing_kernel_results.xlsx')) as writer:
        mean_results_df.to_excel(writer, sheet_name='mean_results')
        std_results_df.to_excel(writer, sheet_name='std_results')
        median_results_df.to_excel(writer, sheet_name='median_results')

if __name__ == '__main__':
    path_1 = sys.argv[1]
    extract_results(path_1)

