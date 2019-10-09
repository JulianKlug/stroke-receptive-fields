import numpy as np
import torch
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]


def compare_metric(result_path1, result_path2, metric):
    results1 = torch.load(result_path1)
    results2 = torch.load(result_path2)

    roc_auc_1 = flatten(results1[metric])
    roc_auc_2 = flatten(results2[metric])

    t, p = mannwhitneyu(roc_auc_1, roc_auc_2)

    print(len(roc_auc_1), len(roc_auc_2))
    print(t, p)


compare_metric('/Users/julian/stroke_research/all_2016_2017_results/subgroup_analysis/non_recanalised_subgroup_analysis/non_recanalised_norm_Tmax0_imagewise_rf_0_output/scores_non_recanalised_norm_Tmax0_imagewise_rf_0.npy',
               '/Users/julian/stroke_research/all_2016_2017_results/subgroup_analysis/recanalised_subgroup_analysis/recanalised_norm_Tmax0_imagewise_rf_0_output/scores_recanalised_norm_Tmax0_imagewise_rf_0.npy',
               'test_image_wise_roc_auc')
