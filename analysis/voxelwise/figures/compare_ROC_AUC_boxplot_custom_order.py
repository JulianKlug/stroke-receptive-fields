import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

def flatten(l):
    if not (type(l[0]) == list or isinstance(l[0], np.ndarray)):
        return l
    return [item for sublist in l for item in sublist]

columns = ['model_name', 'rf', 'model_class', 'model_tag', 'roc_auc', 'dice']
def wrapper_plot_auc_roc_boxplot(modality_dirs, save_dir, order):
    scores_df = pd.DataFrame(columns = columns)

    for modality_dir in modality_dirs[:]:
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

                if 'all_pCT' in model_name:
                    model_class = 'glm(all channels)'
                if 'Tmax' in model_name:
                    model_class = 'glm(Tmax)'
                if 'Tmax0_6' in model_name:
                    model_class = 'Tmax > 6s'
                if 'Tmax0_custom' in model_name:
                    # this should be the same as glm(Tmax) at rf0
                    model_class = 'Tmax > t'
                    continue
                if 'continuous_Tmax' in model_name:
                    model_class = 'g(Tmax)'
                if 'CBV' in model_name:
                    model_class = 'glm(CBV)'
                if 'norm_CBV' in model_name:
                    model_class = 'glm(nCBV)'
                if 'CBF' in model_name:
                    model_class = 'glm(CBF)'
                if 'norm_CBF' in model_name:
                    model_class = 'glm(nCBF)'
                if 'MTT' in model_name:
                    model_class = 'glm(MTT)'
                if 'Campbell' in model_name and 'trained' in model_name:
                    model_class = 'relCBF < t'
                if 'Campbell' in model_name and not 'trained' in model_name:
                    model_class = 'nCBF'

                model_tag = model_class + ' at rf ' + str(rf)

                # leave out other models
                if not (rf == 0 or rf == 3):
                    continue
                test_roc_auc = flatten(results['test_roc_auc'])
                test_dice = flatten(results['test_image_wise_dice'])

                new_entries = list(zip(
                    np.repeat(model_name,len(test_roc_auc)),
                    np.repeat(rf,len(test_roc_auc)),
                    np.repeat(model_class,len(test_roc_auc)),
                    np.repeat(model_tag,len(test_roc_auc)),
                    test_roc_auc,
                    test_dice
                    ))
                scores_df = scores_df.append(pd.DataFrame(new_entries, columns = columns))


    # fig, ax = plt.subplots()
    # g = sns.FacetGrid(scores_df, col='model_class')
    # g.map(sns.boxplot, "model_tag", "roc_auc", hue="rf")
    # g.add_legend()
    # sns.set(font_scale=1.1)
    g = sns.catplot(x="rf", y="roc_auc", hue="rf", col='model_class', dodge=False, aspect=0.6,
                    data=scores_df, kind="box", legend=True, legend_out=True, palette="Pastel2",
                    col_order=['glm(MTT)', 'glm(CBV)', 'glm(CBF)', 'glm(Tmax)', 'glm(all channels)'])
    (g.set_axis_labels("", "Area under the ROC curve")
     .set_xticklabels(["No receptive field", "Receptive field at rf 3"],
                      rotation=35, ha='right')
     .set_titles("{col_name}", size=16)
    )

    g.fig.subplots_adjust(top=0.88)
    plt.suptitle('Boxplots of the area under the ROC curve by model', y=1.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fancybox=True, title="rf")
    plt.show()
    g.fig.savefig(os.path.join(save_dir, "roc_auc_boxplot_per_model.svg"), format="svg")

    # g.add_legend()
    # axes = sns.boxplot(x='model_tag', y="roc_auc", data=scores_df, hue='rf', order=order,
    #                    ax=ax, dodge=False, palette="Pastel2")
    # labels = [l.get_text() for l in axes.get_xticklabels()]
    # ax.set_xticklabels(labels, rotation=35, ha='right')
    # ax.set_title("Boxplots of the area under the ROC curve by model")
    # plt.ylabel('Area under the ROC curve')
    # plt.xlabel('')
    # Receptive field size (as voxels from center)
    # fig.tight_layout()
    # fig1 = plt.gcf()
    # fig1.savefig(os.path.join(save_dir, "roc_auc_boxplot_all.svg"), format="svg")

    # fig, ax = plt.subplots()
    # # axes = sns.boxplot(x = 'model_tag', y="dice", order = order, data = scores_df, hue = 'rf',
    # #                     ax = ax, dodge = False, palette="Pastel2")
    # axes = sns.boxplot(x = 'model_tag', y="dice", data = scores_df, hue = 'rf',
    #                     ax = ax, dodge = False, palette="Pastel2")
    # labels = [l.get_text() for l in axes.get_xticklabels()]
    # ax.set_xticklabels(labels, rotation=35, ha='right')
    # ax.set_title("Boxplots of Dice coefficients by model")
    # plt.ylabel('Sørensen–Dice coefficient')
    # plt.xlabel('')
    # # Receptive field size (as voxels from center)
    # fig.tight_layout()
    # fig1 = plt.gcf()
    # plt.show()
    # fig1.savefig(os.path.join(save_dir, "dice_boxplot_all.svg"), format="svg")

main_dir = '/Users/julian/temp/selected_models_for_ESO_presentation'
multiGLM = os.path.join(main_dir, 'all_pCT_logReg')
MTT = os.path.join(main_dir, 'MTT2_logReg')
Tmax = os.path.join(main_dir, 'Tmax0_logReg')
CBF = os.path.join(main_dir, 'CBF1_logReg')
CBV = os.path.join(main_dir, 'CBV3_logReg')

order = ['glm(MTT) at rf 0', 'glm(MTT) at rf 3',
         'glm(CBV) at rf 0', 'glm(CBV) at rf 3',
         'glm(CBF) at rf 0', 'glm(CBF) at rf 3',
         'glm(Tmax) at rf 0', 'glm(Tmax) at rf 3',
         'glm(multi) at rf 0', 'glm(multi) at rf 3',
         ]

wrapper_plot_auc_roc_boxplot([multiGLM, MTT, Tmax, CBF, CBV], main_dir, order)
