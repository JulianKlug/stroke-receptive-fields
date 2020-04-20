import brain_mask as bm

data_dir = '/home/klug/data/original_data/with_angio_2016_2017/nifti_all_angio'
bm.createBrainMaskWrapper(data_dir, restrict_to_RAPID_maps=False, high_resolution = False)
