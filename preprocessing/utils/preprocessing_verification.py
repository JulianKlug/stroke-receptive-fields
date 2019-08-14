import os, sys
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from naming_verification import loose_verify_name
import numpy as np


def visual_verification(data_dir):
    subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, o))]

    sequences = ['wreor_SPC', 'wcoreg_CBF', 'wcoreg_t2', 'wcoreg_VOI']

    plt.switch_backend('agg')
    ncol = 10
    nrow = len(subjects) + 2
    figure = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=1, hspace=0.25,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    i_subj = 0
    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, o))]

        visual_add(np.empty((3,3,3)), i_subj, 0, gs, subject)
        i_image = 1

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                       if o.endswith(".nii")]

            for study in studies:
                study_path = os.path.join(modality_dir, study)

                if loose_verify_name(study, sequences):
                    img = nib.load(study_path)
                    data = img.get_data()
                    tag = '-'.join(study.split('-')[0].split('_')[1:-1])
                    visual_add(data, i_subj, i_image, gs, tag)
                    i_image += 1

        i_subj += 1

    plt.ioff()
    plt.switch_backend('agg')
    figure_path = os.path.join(data_dir,  'processing_visualisation.svg')
    figure.savefig(figure_path, dpi='figure', format='svg')
    plt.close(figure)

# draw image on canvas
def visual_add(image, i_subj, i_image, gs, image_id=None):
    n_z = image.shape[2]
    center_z = (n_z - 1) // 2
    i_col = i_image
    i_row = i_subj

    # plot image
    ax = plt.subplot(gs[i_row, i_col])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(-image[:, :, center_z].T)
    plt.set_cmap('Greys')
    plt.axis('off')


if __name__ == '__main__':
    path = sys.argv[1]
    visual_verification(path)
