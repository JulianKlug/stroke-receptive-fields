import sys, torch, timeit
sys.path.insert(0, '../')
import numpy as np
import data_loader
from vxl_glm.LogReg_glm import LogReg_glm
from vxl_continuous.normalized_marker_model import Normalized_marker_Model_Generator
from utils import rescale_outliers, standardise
from voxelwise import receptiveField
from visual import display


def model_predict(model_path, data_dir, model_type, subject_id, rf=3, fold=0, inverse_relation=False,
                  feature_scaling=True, channel=None):
    # Load data
    clinical_inputs, ct_inputs, ct_label, _, _, brain_masks, ids, params = data_loader.load_saved_data(data_dir)
    # Order: 'wcoreg_RAPID_Tmax', 'wcoreg_RAPID_rCBF', 'wcoreg_RAPID_MTT', 'wcoreg_RAPID_rCBV'
    if channel is not None:
        print('Using only channel:', ['Tmax', 'rCBF', 'MTT', 'rCBV'][channel])
        ct_inputs = ct_inputs[:, :, :, :, channel]
        ct_inputs = np.expand_dims(ct_inputs, axis=5)
    n_subj, n_x, n_y, n_z, n_c = ct_inputs.shape

    subj_index = np.argwhere(ids == subject_id)
    if len(subj_index) < 1:
        raise Exception('Subject not found:', subject_id)
    else: subj_index = subj_index[0, 0]

    print(ct_inputs.shape, subj_index)

    imgX = np.expand_dims(ct_inputs[subj_index], axis=0)
    maskX = np.expand_dims(brain_masks[subj_index], axis=0)
    print(imgX.shape)

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

    threeD_probas = np.full(maskX.shape, 0, dtype=np.float64)
    threeD_probas[maskX] = probas.reshape(maskX.shape)[maskX]


    print('Prediction in', end - start, 'seconds')
    display(np.squeeze(threeD_probas), cmap='jet', block=False)
    display(ct_label[subj_index])







