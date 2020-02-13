import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import os

def get_model_title(model_name):
    if 'CBF' in model_name: return 'g(CBF)'
    if 'CBV' in model_name: return 'g(CBV)'
    if 'MTT' in model_name: return 'g(MTT)'
    if 'Tmax' in model_name: return 'g(Tmax)'
    else:
        return 'other'


def AUC_per_kernel_width(smoothing_kernel_results_file, save=True, legend=True):
    smoothing_kernel_results_df = pd.read_excel(smoothing_kernel_results_file)
    data_name = os.path.basename(smoothing_kernel_results_file)

    if '2D' in data_name:
        dimension = '2D'
    elif '3D' in data_name:
        dimension = '3D'
    else:
        raise Exception('dimension should be in data file title')

    if save:
        plt.switch_backend('agg')
    ncol = 4
    nrow = 1
    figure = plt.figure(figsize=(6 * ncol + 1, 4 * nrow + 1))
    figure.suptitle('Area under the ROC curve for normalized models after increasing widths of ' + dimension + ' smoothing kernels')
    gs = gridspec.GridSpec(nrow, ncol, hspace=0.3)

    # unify model names
    smoothing_kernel_results_df['model'] = smoothing_kernel_results_df['model'].apply(lambda x: x.split('_k')[0])
    # transform voxel to mm
    smoothing_kernel_results_df['kernel_width'] = smoothing_kernel_results_df['kernel_width'].apply(lambda x: 2*x)

    per_model = smoothing_kernel_results_df.groupby('model')

    m_index = 0
    for model_name, model_performance_df in per_model:
        i_line = (m_index // gs.get_geometry()[1])
        i_row = m_index % gs.get_geometry()[1]

        ax = plt.subplot(gs[i_line, i_row])
        model_performance_df.plot.scatter(x='kernel_width',
                                          y='test_roc_auc',
                                          c='DarkBlue',
                                          ax=ax,
                                          title=get_model_title(model_name))
        ax.set_xlabel('Kernel width (mm)')
        ax.set_ylabel('AUC')
        ax.set_ylim(0.5, 0.9)
        m_index += 1

    if legend:
        legend_elements = []
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Mean AUC',
                                      markerfacecolor='DarkBlue', markersize=6))

        plt.figlegend(handles=legend_elements, loc='right', fontsize='large')

    if save:
        # plt.switch_backend('agg')
        data_dir = os.path.dirname(smoothing_kernel_results_file)
        plt.savefig(os.path.join(data_dir, data_name.split('.')[0] + '_AUC.svg'), format="svg")

    plt.show()

    # print(smoothing_kernel_results_df.groupby('model')['test_roc_auc'].mean())

AUC_per_kernel_width('/Users/julian/OneDrive - unige.ch/stroke_research/rf_supplementary_analyses/2d_vs_3d_smoothing_continuous_models/2D_smoothing_kernel_results.xlsx')


