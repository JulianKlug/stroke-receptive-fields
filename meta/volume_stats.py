import sys
sys.path.insert(0, '../')
import numpy as np
import pandas as pd
from analysis import data_loader
from analysis.subgroup_analysis import IAT_status_split


def VOI_lesion_volume_stats(data_dir, IAT_status_path=None):
    data_set = data_loader.load_saved_data(data_dir)
    clinical_inputs, ct_inputs, ct_label, _, _, brain_masks, ids, params = data_set
    volumes_in_vx = np.sum(ct_label, axis=(1, 2, 3))
    volumes_in_mm3 = volumes_in_vx * 8

    print('Overall: GT label volumes')
    print('-- median:', np.median(volumes_in_mm3))
    print('-- Q25:', np.percentile(volumes_in_mm3, 25))
    print('-- Q75', np.percentile(volumes_in_mm3, 75))

    if IAT_status_path is not None:
        IAT_indexes, IVT_indexes, unknown_status_indexes = IAT_status_split.split_dataset(data_set, IAT_status_path)
        print(len(unknown_status_indexes), 'subjects with unknown status:', ids[unknown_status_indexes])

        iat_volumes_in_vx = np.sum(ct_label[IAT_indexes], axis=(1, 2, 3))
        iat_volumes_in_mm3 = iat_volumes_in_vx * 8

        print('IAT: GT label volumes [mm3]')
        print('-- median:', np.median(iat_volumes_in_mm3))
        print('-- Q25:', np.percentile(iat_volumes_in_mm3, 25))
        print('-- Q75', np.percentile(iat_volumes_in_mm3, 75))

        ivt_volumes_in_vx = np.sum(ct_label[IVT_indexes], axis=(1, 2, 3))
        ivt_volumes_in_mm3 = ivt_volumes_in_vx * 8

        print('IVT: GT label volumes [mm3]')
        print('-- median:', np.median(ivt_volumes_in_mm3))
        print('-- Q25:', np.percentile(ivt_volumes_in_mm3, 25))
        print('-- Q75', np.percentile(ivt_volumes_in_mm3, 75))

def RAPID_volume_stats(RAPID_volume_file_path, IAT_status_path=None, data_dir=None):
    RAPID_volume_df = pd.read_excel(RAPID_volume_file_path)
    # Transform ml to mm3
    RAPID_volume_df['RAPID_CBF30_volume_ml'] = RAPID_volume_df['RAPID_CBF30_volume_ml'].apply(lambda x: x * 1000)
    RAPID_volume_df['RAPID_Tmax6_volume_ml'] = RAPID_volume_df['RAPID_Tmax6_volume_ml'].apply(lambda x: x * 1000)

    print('Overall: RAPID ischemic core volumes [mm3]')
    print('-- median:', RAPID_volume_df.median()['RAPID_CBF30_volume_ml'])
    print('-- Q25:', RAPID_volume_df.quantile(0.25)['RAPID_CBF30_volume_ml'])
    print('-- Q75', RAPID_volume_df.quantile(0.75)['RAPID_CBF30_volume_ml'])

    print('Overall: RAPID penumbra volumes [mm3]')
    print('-- median:', RAPID_volume_df.median()['RAPID_Tmax6_volume_ml'])
    print('-- Q25:', RAPID_volume_df.quantile(0.25)['RAPID_Tmax6_volume_ml'])
    print('-- Q75', RAPID_volume_df.quantile(0.75)['RAPID_Tmax6_volume_ml'])

    if IAT_status_path is not None and data_dir is not None:
        data_set = data_loader.load_saved_data(data_dir)
        clinical_inputs, ct_inputs, ct_label, _, _, brain_masks, ids, params = data_set
        IAT_indexes, IVT_indexes, unknown_status_indexes = IAT_status_split.split_dataset(data_set, IAT_status_path)
        print(len(unknown_status_indexes), 'subjects with unknown status:', ids[unknown_status_indexes])

        IAT_ids = ids[IAT_indexes]
        IAT_RAPID_volume_df = RAPID_volume_df[RAPID_volume_df.anonymised_id.isin(IAT_ids)]

        print('IAT: RAPID ischemic core volumes [mm3]')
        print('-- median:', IAT_RAPID_volume_df.median()['RAPID_CBF30_volume_ml'])
        print('-- Q25:', IAT_RAPID_volume_df.quantile(0.25)['RAPID_CBF30_volume_ml'])
        print('-- Q75', IAT_RAPID_volume_df.quantile(0.75)['RAPID_CBF30_volume_ml'])

        print('IAT: RAPID penumbra volumes [mm3]')
        print('-- median:', IAT_RAPID_volume_df.median()['RAPID_Tmax6_volume_ml'])
        print('-- Q25:', IAT_RAPID_volume_df.quantile(0.25)['RAPID_Tmax6_volume_ml'])
        print('-- Q75', IAT_RAPID_volume_df.quantile(0.75)['RAPID_Tmax6_volume_ml'])

        IVT_ids = ids[IVT_indexes]
        IVT_RAPID_volume_df = RAPID_volume_df[RAPID_volume_df.anonymised_id.isin(IVT_ids)]

        print('IVT: RAPID ischemic core volumes [mm3]')
        print('-- median:', IVT_RAPID_volume_df.median()['RAPID_CBF30_volume_ml'])
        print('-- Q25:', IVT_RAPID_volume_df.quantile(0.25)['RAPID_CBF30_volume_ml'])
        print('-- Q75', IVT_RAPID_volume_df.quantile(0.75)['RAPID_CBF30_volume_ml'])

        print('IVT: RAPID penumbra volumes [mm3]')
        print('-- median:', IVT_RAPID_volume_df.median()['RAPID_Tmax6_volume_ml'])
        print('-- Q25:', IVT_RAPID_volume_df.quantile(0.25)['RAPID_Tmax6_volume_ml'])
        print('-- Q75', IVT_RAPID_volume_df.quantile(0.75)['RAPID_Tmax6_volume_ml'])







# VOI_lesion_volume_stats('/Users/julian/stroke_research/data/all2016_subset_prepro',
#                        '/Users/julian/OneDrive - unige.ch/stroke_research/rf_article/JCBFM_submission/data/clinical_data_all_2016_2017/included_subjects_data_23072019/recanalisation_status.xlsx')

RAPID_volume_stats('/Users/julian/OneDrive - unige.ch/stroke_research/image_metadata/RAPID_volumes.xlsx',
                    IAT_status_path='/Users/julian/OneDrive - unige.ch/stroke_research/rf_article/JCBFM_submission/data/clinical_data_all_2016_2017/included_subjects_data_23072019/recanalisation_status.xlsx',
                    data_dir= '/Users/julian/stroke_research/data/all2016_subset_prepro')

