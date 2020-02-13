import os, torch, sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

columns = ['model', 'test_roc_auc', 'test_image_wise_dice', 'test_image_wise_hausdorff',
            'test_accuracy', 'test_f1', 'test_jaccard', 'test_thresholded_volume_deltas',
           'test_unthresholded_volume_deltas', 'test_image_wise_error_ratios', 'evaluation_thresholds', 'test_TPR',
           'test_FPR']

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]

def compare_results(score_file_1, score_file_2, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(score_file_1)

    results_1 = torch.load(score_file_1)
    model_1 = os.path.basename(score_file_1).split('.')[0]

    results_2 = torch.load(score_file_2)
    model_2 = os.path.basename(score_file_2).split('.')[0]

    models = [(model_1, results_1), (model_2, results_2)]

    for model_name, results in models:

        mean_list = [[model_name] + [np.mean(flatten(results[i])) if i in results else np.nan for i in columns[1:]]]
        std_list = [[model_name] + [np.std(flatten(results[i])) if i in results else np.nan for i in columns[1:]]]
        median_list = [[model_name] + [np.median(flatten(results[i])) if i in results else np.nan for i in columns[1:]]]

        # print(np.array([np.array(results[i]) if i in results else np.repeat(np.nan, 50).reshape(1,50) for i in columns[2:-1]]).shape)
        # print(np.repeat(np.nan, 50).reshape(1,50).shape)
        # print(np.array([np.array(results[i]) if i in results else np.repeat(np.nan, 50).reshape(1,50) for i in columns[2:]]).shape)
        n_runs = len(results[columns[2]])
        current_all_list = np.concatenate((
            np.repeat(model_name, n_runs).reshape(1, n_runs),
            [np.squeeze(np.array(results[i])) if i in results else np.repeat(np.nan, n_runs) for i in columns[1:]]
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
            else:
                all_list[selector] = all_list[selector].astype(float)

        try:
            all_results_array
        except NameError:
            all_results_array = np.expand_dims(np.array(all_list), axis=0)
        else:
            all_results_array = np.concatenate((all_results_array,
                                                np.expand_dims(np.array(all_list), axis=0)),
                                               axis=0)
        try:
            std_results_array
        except NameError:
            std_results_array = np.array(std_list)
        else:
            std_results_array = np.concatenate((std_results_array,
                                                np.array(std_list)))

        try:
            mean_results_array
        except NameError:
            mean_results_array = np.array(mean_list)
        else:
            mean_results_array = np.concatenate((mean_results_array,
                                                 np.array(mean_list)))

        try:
            median_results_array
        except NameError:
            median_results_array = np.array(median_list)
        else:
            median_results_array = np.concatenate((median_results_array,
                                                   np.array(median_list)))

    mean_results_df = pd.DataFrame(mean_results_array, columns=columns)
    std_results_df = pd.DataFrame(std_results_array, columns=columns)
    median_results_df = pd.DataFrame(median_results_array, columns=columns)


    # Comparison
    model_1_array = np.squeeze(np.array([k for k in all_results_array if k[0,0] == model_1]))
    model_2_array = np.squeeze(np.array([k for k in all_results_array if k[0,0] == model_2]))

    p_val_array = []
    for compared_column_index in range(1, len(columns)):
        print('Comparing', columns[compared_column_index])
        try:
            t, p = wilcoxon(flatten(model_1_array[compared_column_index]),
                            flatten(model_2_array[compared_column_index]))
        except:
            p = np.nan
        p_val_array.append(p)

    compared_results_df = pd.DataFrame([np.concatenate((['comparison'], np.array(p_val_array)))], columns=columns)


    with pd.ExcelWriter(os.path.join(output_dir, str(model_1 + '_vs_' + model_2 + '_comparison.xlsx'))) as writer:
        mean_results_df.to_excel(writer, sheet_name='mean_results')
        median_results_df.to_excel(writer, sheet_name='median_results')
        std_results_df.to_excel(writer, sheet_name='std_results')
        compared_results_df.to_excel(writer, sheet_name='p_vals')

if __name__ == '__main__':
    path_1 = sys.argv[1]
    path_2 = sys.argv[2]

    compare_results(path_1, path_2)



