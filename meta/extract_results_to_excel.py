import os, torch
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

main_dir = '/Users/julian/master/server_output/selected_for_article1_13022019'
output_dir = '/Users/julian/master/saved_results'

columns = ['model','rf', 'test_roc_auc', 'test_image_wise_dice', 'test_image_wise_hausdorff',
'test_accuracy', 'test_f1', 'test_jaccard', 'test_thresholded_volume_deltas', 'test_unthresholded_volume_deltas', 'test_image_wise_error_ratios', 'evaluation_thresholds']

modalites = os.listdir(main_dir)

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]


for modality in modalites:
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
                rf = int(np.median(params['rf']))
            except (KeyError, TypeError):
                rf = int(result_files[0].split('_')[-1].split('.')[0])

            median_list = [[model_name, rf] + [np.median(flatten(results[i])) if i in results else np.nan for i in columns[2:]]]

            # print(np.array([np.array(results[i]) if i in results else np.repeat(np.nan, 50).reshape(1,50) for i in columns[2:-1]]).shape)
            # print(np.repeat(np.nan, 50).reshape(1,50).shape)
            # print(np.array([np.array(results[i]) if i in results else np.repeat(np.nan, 50).reshape(1,50) for i in columns[2:]]).shape)
            all_list = np.concatenate((
                np.repeat(model_name, 50).reshape(1, 50),
                np.repeat(rf, 50).reshape(1,50),
                [np.array(results[i]) if i in results else np.repeat(np.nan, 50) for i in columns[2:]]
                ))

            try:
                all_results_array
            except NameError:
                all_results_array = np.expand_dims(np.array(all_list), axis=0)
            else :
                all_results_array = np.concatenate((all_results_array,
                                np.expand_dims(np.array(all_list), axis=0)),
                                axis = 0)

            try:
                median_results_array
            except NameError:
                median_results_array = np.array(median_list)
            else :
                median_results_array = np.concatenate((median_results_array,
                                np.array(median_list)))

median_results_df = pd.DataFrame(median_results_array, columns = columns)

# Compare rf3 to rf0 for the same model
rf_comp_df_columns = ['model','median_rf0', 'median_rf3', 'Pval', 'compared_variable']
rf_3_model_results = np.array([k for k in all_results_array if k[1,0] == 3])
for rf_3_model_result in rf_3_model_results:
    model_base = '_'.join(rf_3_model_result[0, 0].split('_')[:-1])
    corres_rf0_model = [k for k in all_results_array if k[1,0] == 0 and k[0,0].startswith(model_base)][0]

    # compare roc auc
    compared_variable_index = 2
    print('Comparing rf with', columns[compared_variable_index], 'for', model_base)
    t, p = wilcoxon(flatten(rf_3_model_result[compared_variable_index]), flatten(corres_rf0_model[compared_variable_index]))
    comparative_list = [[model_base,
        np.median(flatten(corres_rf0_model[compared_variable_index])),
        np.median(flatten(rf_3_model_result[compared_variable_index])),
        p,
        columns[compared_variable_index]
        ]]

    try:
        comparative_results_array
    except NameError:
        comparative_results_array = np.array(comparative_list)
    else :
        comparative_results_array = np.concatenate((comparative_results_array,
                        np.array(comparative_list)))
rf_comp_results_df = pd.DataFrame(comparative_results_array, columns = rf_comp_df_columns)

# Compare modalities to best (Tmax0_logRegGLM at rf3)
modality_comp_df_columns = ['model','rf', 'model_result', 'ref_model_result', 'Pval', 'compared_variable', 'reference_rf3_model']
reference_rf3_model_results = np.squeeze(np.array([k for k in all_results_array if k[1,0] == 3 and 'Tmax0_logRegGLM' in k[0,0]]))
rf_3_model_results = np.array([k for k in all_results_array if k[1,0] == 3])
for rf_3_model_result in rf_3_model_results:
    model_base = '_'.join(rf_3_model_result[0, 0].split('_')[:-1])

    # compare roc auc
    compared_variable_index = 2
    print('Comparing modality with', columns[compared_variable_index], 'for', rf_3_model_result[0,0])
    t, p = wilcoxon(flatten(rf_3_model_result[compared_variable_index]), flatten(reference_rf3_model_results[compared_variable_index]))
    modality_comp_list = [[model_base,
        rf_3_model_result[1,0],
        np.median(flatten(rf_3_model_result[compared_variable_index])),
        np.median(flatten(reference_rf3_model_results[compared_variable_index])),
        p,
        columns[compared_variable_index],
        reference_rf3_model_results[0,0]
        ]]

    try:
        modality_comp_results_array
    except NameError:
        modality_comp_results_array = np.array(modality_comp_list)
    else :
        modality_comp_results_array = np.concatenate((modality_comp_results_array,
                        np.array(modality_comp_list)))
modality_comp_results_df = pd.DataFrame(modality_comp_results_array, columns = modality_comp_df_columns)



with pd.ExcelWriter(os.path.join(output_dir, 'rf_article_results.xlsx')) as writer:
    median_results_df.to_excel(writer, sheet_name='median_results')
    rf_comp_results_df.to_excel(writer, sheet_name='rf_comparative_results')
    modality_comp_results_df.to_excel(writer, sheet_name='modality_comparative_results')
