import os, torch, math
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

main_dir = '/Users/julian/temp/IAT_subgroup_analysis'
output_dir = main_dir

columns = ['model', 'rf', 'test_roc_auc', 'test_image_wise_dice', 'test_image_wise_hausdorff',
'test_accuracy', 'test_f1', 'test_jaccard', 'test_thresholded_volume_deltas', 'test_unthresholded_volume_deltas', 'test_image_wise_error_ratios', 'evaluation_thresholds']

modalites = os.listdir(main_dir)

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    if isinstance(l[0], np.ndarray):
        l = [sublist.tolist() for sublist in l]
    l = [sublist if (type(sublist) == list or isinstance(sublist, np.ndarray)) else [sublist] for sublist in l]
    return [item for sublist in l for item in sublist]


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

            mean_list = [[model_name, rf] + [np.mean(flatten(results[i])) if i in results else np.nan for i in columns[2:]]]
            std_list = [[model_name, rf] + [np.std(flatten(results[i])) if i in results else np.nan for i in columns[2:]]]
            median_list = [[model_name, rf] + [np.median(flatten(results[i])) if i in results else np.nan for i in columns[2:]]]

            n_runs = len(results[columns[2]])
            current_all_list = np.concatenate((
                np.repeat(model_name, n_runs).reshape(1, n_runs),
                np.repeat(rf, n_runs).reshape(1, n_runs),
                [np.squeeze(np.array(results[i])) if i in results else np.repeat(np.nan, n_runs) for i in columns[2:]]
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

            # For image_wise metrics, make sure all arrays have the same length (needed for further comparison)
            for selector in range(2, all_list.shape[0]):
                if not (type(all_list[selector][0]) == list or isinstance(all_list[selector][0], np.ndarray)):
                    continue
                max_sublist_len = max([fold.size for fold in all_list[selector]])
                for fold_idx in range(all_list[selector].size):
                    fixed_size_fold = np.empty(max_sublist_len)
                    fixed_size_fold[:] = np.nan
                    fixed_size_fold[:all_list[selector][fold_idx].size] = all_list[selector][fold_idx]
                    all_list[selector][fold_idx] = fixed_size_fold

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
            else:
                median_results_array = np.concatenate((median_results_array,
                                                       np.array(median_list)))

mean_results_df = pd.DataFrame(mean_results_array, columns = columns)
std_results_df = pd.DataFrame(std_results_array, columns = columns)
median_results_df = pd.DataFrame(median_results_array, columns=columns)


# Compare all rfs for the same model
all_rf_comp_df_columns = ['model','compared_variable', 'p0-1', 'p1-2', 'p2-3', 'p3-4', 'p4-5']
all_rf_0_model_results = np.array([k for k in all_results_array if k[1,0] == 0])
for all_rf_0_model_result in all_rf_0_model_results:
    # compare roc auc
    compared_variable_index = 2
    model_base = '_'.join(all_rf_0_model_result[0, 0].split('_')[:-1])
    all_rf_comp_list = [model_base, columns[compared_variable_index]]
    for i in range(5):
        ref_model = [k for k in all_results_array if k[1,0] == i and k[0,0].startswith(model_base)]
        corres_rf_plus1_model = [k for k in all_results_array if k[1,0] == i+1 and k[0,0].startswith(model_base)]
        if not ref_model or not corres_rf_plus1_model:
            all_rf_comp_list.append(np.nan)
            continue
        ref_model = ref_model[0]
        corres_rf_plus1_model = corres_rf_plus1_model[0]

        print('Comparing rf with', columns[compared_variable_index], 'for', model_base, 'for', i, 'and', i+1)
        t, p = wilcoxon(flatten(ref_model[compared_variable_index]), flatten(corres_rf_plus1_model[compared_variable_index]))
        all_rf_comp_list.append(p)

    all_rf_comp_list = [all_rf_comp_list]

    try:
        all_rf_comp_array
    except NameError:
        all_rf_comp_array = np.array(all_rf_comp_list)
    else :
        all_rf_comp_array = np.concatenate((all_rf_comp_array,
                        np.array(all_rf_comp_list)))
all_rf_comp_results_df = pd.DataFrame(all_rf_comp_array, columns = all_rf_comp_df_columns)

# Compare rf3 to rf0 for the same model
# 2 --> compare roc auc
compared_variable_index = 3
rf_comp_df_columns = ['model','mean_rf0', 'mean_rf3', 'Pval', 'compared_variable']
rf_3_model_results = np.array([k for k in all_results_array if k[1,0] == 3])
all_rf0 = []
all_rf3 = []
for rf_3_model_result in rf_3_model_results:
    model_base = '_'.join(rf_3_model_result[0, 0].split('_')[:-1])
    corres_rf0_model = [k for k in all_results_array if k[1,0] == 0 and k[0,0].startswith(model_base)][0]

    print('Comparing rf with', columns[compared_variable_index], 'for', model_base)
    t, p = wilcoxon(flatten(rf_3_model_result[compared_variable_index]), flatten(corres_rf0_model[compared_variable_index]))
    all_rf3.append(flatten(rf_3_model_result[compared_variable_index]))
    all_rf0.append(flatten(corres_rf0_model[compared_variable_index]))
    comparative_list = [[model_base,
        np.nanmean(flatten(corres_rf0_model[compared_variable_index])),
        np.nanmean(flatten(rf_3_model_result[compared_variable_index])),
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

t, p = wilcoxon(flatten(all_rf0), flatten(all_rf3))
comparative_results_array = np.concatenate((comparative_results_array,
                np.array([[
                'mean_model',
                np.nanmean(flatten(all_rf0)),
                np.nanmean(flatten(all_rf3)),
                p,
                columns[compared_variable_index]
                ]])))
rf_comp_results_df = pd.DataFrame(comparative_results_array, columns = rf_comp_df_columns)

# for k in all_results_array:
#     print(k[0, 0], k[1,0], type(k[1,0]), int(k[1,0]) == 3, k[2,0], type(k[2,0]))

# Compare modalities to best (Tmax0_logRegGLM at rf3)
modality_comp_df_columns = ['model','rf', 'model_result', 'ref_model_result', 'Pval', 'compared_variable', 'reference_rf3_model']
reference_rf3_model_results = np.squeeze(np.array([k for k in all_results_array if k[1,0] == 3 and 'IVT_Tmax0_GLM' in k[0,0]]))
print('Using', reference_rf3_model_results[0,0], 'as reference for comparison')
rf_3_model_results = np.array([k for k in all_results_array if k[1,0] == 3])
# rf_3_model_results = np.array([k for k in all_results_array])

for rf_3_model_result in rf_3_model_results:
    model_base = '_'.join(rf_3_model_result[0, 0].split('_')[:-1])

    # 2 --> compare roc auc
    compared_variable_index = 3
    print('Comparing modality with', columns[compared_variable_index], 'for', rf_3_model_result[0,0])
    # t, p = wilcoxon(flatten(rf_3_model_result[compared_variable_index]), flatten(reference_rf3_model_results[compared_variable_index]))

    # For separate populations
    t, p = mannwhitneyu(flatten(rf_3_model_result[compared_variable_index]), flatten(reference_rf3_model_results[compared_variable_index]))
    modality_comp_list = [[model_base,
        rf_3_model_result[1,0],
        np.nanmean(flatten(rf_3_model_result[compared_variable_index])),
        np.nanmean(flatten(reference_rf3_model_results[compared_variable_index])),
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



with pd.ExcelWriter(os.path.join(output_dir, 'rf_mean_article_results.xlsx')) as writer:
    mean_results_df.to_excel(writer, sheet_name='mean_results')
    median_results_df.to_excel(writer, sheet_name='median_results')
    std_results_df.to_excel(writer, sheet_name='std_results')
    rf_comp_results_df.to_excel(writer, sheet_name='0-3_rf_comparative_results')
    all_rf_comp_results_df.to_excel(writer, sheet_name='all_rf_comparative_results')
    modality_comp_results_df.to_excel(writer, sheet_name='modality_comparative_results')
