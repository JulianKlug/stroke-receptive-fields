import sys, torch, timeit, os
sys.path.insert(0, '../')
import numpy as np
import data_loader
from vxl_glm.LogReg_glm import LogReg_glm
from vxl_continuous.normalized_marker_model import Normalized_marker_Model_Generator
from utils import rescale_outliers, standardise
from voxelwise import receptiveField
from visual import display
import skimage.measure as measure

def create_background_mask(ct_inputs):
    channel_masks = []
    for c in range(ct_inputs.shape[-1]):
        labeled, n_labels = measure.label(ct_inputs[ ..., c], background=-1, return_num=True)
        labeled = labeled == 1
        labeled = labeled.astype(int)
        channel_masks.append(labeled)
    combined_labels = np.array(channel_masks).sum(axis=0)
    combined_labels = -1 * (combined_labels > 3) + 1
    labeled = combined_labels.astype(int)
    return labeled

def model_predict(model_path, data_dir, model_type, subject_id, rf=3, fold=0, inverse_relation=False,
                  feature_scaling=True, channel=None, mask_CSF=True, mask_background=True, save_dir=None, save_GT=False):
    # Load data
    clinical_inputs, ct_inputs, ct_label, _, _, brain_masks, ids, params = data_loader.load_saved_data(data_dir)
    # Order: 'wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV'

    subj_index = np.argwhere(ids == subject_id)
    if len(subj_index) < 1:
        raise Exception('Subject not found:', subject_id)
    else: subj_index = subj_index[0, 0]

    if not mask_CSF and mask_background:
        background_mask = np.expand_dims(create_background_mask(ct_inputs[subj_index]), axis=0)

    if channel is not None:
        print('Using only channel:', ['Tmax', 'rCBF', 'MTT', 'rCBV'][channel])
        ct_inputs = ct_inputs[:, :, :, :, channel]
        ct_inputs = np.expand_dims(ct_inputs, axis=5)
    n_subj, n_x, n_y, n_z, n_c = ct_inputs.shape

    imgX = np.expand_dims(ct_inputs[subj_index], axis=0)
    maskX = np.expand_dims(brain_masks[subj_index], axis=0)

    # Load model
    saved_model = torch.load(model_path)[fold]

    if (model_type == 'glm'):
        Model_Generator = LogReg_glm
    elif model_type == 'norm':
        Model_Generator = Normalized_marker_Model_Generator(np.squeeze(ct_inputs).shape, feature_scaling=feature_scaling,
                                                            inverse_relation=inverse_relation)
    else:
        raise Exception('Model not known. Must be one of: glm, norm')

    Model_Generator.hello_world()
    model = Model_Generator('', fold, n_channels=n_c, n_channels_out=1, rf=rf, model=saved_model)

    # Preprocess data
    # rescale outliers
    imgX = rescale_outliers(imgX, MASKS=maskX)

    if feature_scaling:
        # Standardise data (data - mean / std)
        if feature_scaling:
            imgX, _ = standardise(imgX, None)

    start = timeit.default_timer()

    # transform to receptive fields
    rf_inputs, rf_outputs = receptiveField.reshape_to_receptive_field(imgX, np.empty(maskX.shape), (rf, rf, rf))

    probas = model.predict(rf_inputs, np.ones(maskX.shape))

    end = timeit.default_timer()

    threeD_probas = probas.reshape(maskX.shape)
    if mask_CSF:
        threeD_probas[maskX != 1] = 0
    elif mask_background:
        threeD_probas[background_mask != 1] = 0


    print('Prediction in', end - start, 'seconds')
    figure = display(np.squeeze(threeD_probas), cmap='jet', block=False)
    if save_dir is not None:
        figure_name = os.path.basename(model_path).split('.')[0].split('trained_models_')[1] \
                      + '_fold' + str(fold) + '_' + subject_id
        if mask_CSF:
            figure_name += '_masked_CSF'
        elif mask_background:
            figure_name += '_masked_background'
        figure.savefig(os.path.join(save_dir, figure_name + '.png'), dpi='figure', format="png")
    if save_dir is not None and save_GT:
        gt_figure = display(ct_label[subj_index], block=False)
        gt_figure.savefig(os.path.join(save_dir, subject_id + '_GT.png'), dpi='figure', format="png")
    else:
        display(ct_label[subj_index])











