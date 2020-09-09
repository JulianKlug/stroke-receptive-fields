import sys, os
sys.path.insert(0, '../')
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def visualize_dataset(data, channel_names, subject_ids, save_dir, save_name='dataset_visualisation'):
    '''
    Display dataset
    :param data: image data of dimensions (n_subj, x, y, z, c)
    :param channel_names: list of names of the c channels (c) eg.: ['Tmax', 'CBF', 'MTT', 'CBV']
    :return: Save a visualisation file in the save_dir
    '''

    n_subj, _, _, _, n_c = data.shape
    if not n_c == len(channel_names):
        raise Exception(f'Please provide names for all channels; {n_c} vs {len(channel_names)}')

    plt.switch_backend('agg')
    ncol = n_c + 1
    nrow = n_subj + 2
    figure = plt.figure(figsize=(2*(ncol+1) + 1, 2*(nrow+1)))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=1, hspace=0.25,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    # Add Header for data column names
    # visual_add_center_slice(np.zeros((2, 2, 2)), 0, 0, gs, 'Subject-ID')
    for c_idx, channel_name in enumerate(channel_names):
        visual_add_center_slice(np.empty((2, 2, 2)), 0, c_idx + 1, gs, channel_name)

    for subj in tqdm(range(n_subj)):
        subj_data = data[subj]

        visual_add_center_slice(np.random.randint(2, size=(3, 3, 3)), subj, 0, gs, subject_ids[subj])

        for channel in range(n_c):
            visual_add_center_slice(subj_data[..., channel], subj, channel + 1, gs, image_id=None)

    plt.ioff()
    plt.switch_backend('agg')
    figure_path = os.path.join(save_dir, save_name + '.png')
    figure.savefig(figure_path, dpi='figure', format='png')
    plt.close(figure)


# draw center image on canvas
def visual_add_center_slice(image, i_subj, i_image, gs, image_id=None):
    n_z = image.shape[2]
    center_z = (n_z - 1) // 2
    i_col = i_image
    i_row = i_subj

    # plot image
    ax = plt.subplot(gs[i_row, i_col])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(image[:, :, center_z].T, cmap='gray')
    plt.axis('off')

