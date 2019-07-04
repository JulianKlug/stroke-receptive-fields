import os, hashlib
import pandas as pd
import numpy as np
from unidecode import unidecode
from difflib import get_close_matches, SequenceMatcher

def extract_patient_characteristics(patient_id_path, patient_info_path, id_sheet = 'Sheet1', info_sheet = 'Sheet1'):
    """
    """
    # Load spreadsheet
    patient_info_xl = pd.ExcelFile(patient_info_path)
    patient_id_xl = pd.ExcelFile(patient_id_path)
    # Load a sheet into a DataFrame
    patient_info_df = patient_info_xl.parse(info_sheet)
    patient_id_df = patient_id_xl.parse(id_sheet)

    patient_info_df['combined_id'] = patient_info_df['Nom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                    + '^' + patient_info_df['Prénom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                   + '^' + patient_info_df['birth_date'].astype(str).str.split("-").str.join('')
    patient_info_df['hashed_id'] = ['subj-' + str(hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:8]) for item in patient_info_df['combined_id']]

    match_list = [get_close_matches(item, patient_info_df['combined_id'], 1) for item in patient_id_df['patient_identifier']]
    patient_id_df['combined_id'] = [matches[0] if matches else np.NAN for matches in match_list]
    patient_id_df['combined_id_match'] = [SequenceMatcher(None, row['patient_identifier'], row['combined_id']).ratio()
                                          if not row.isnull().values.any() else np.nan
                                          for index, row in patient_id_df.iterrows()]

    output_df = patient_id_df.merge(patient_info_df, how='left', left_on='combined_id', right_on='combined_id')
    output_df['time_to_ct'] = pd.to_datetime(output_df['ct_date']) - pd.to_datetime(output_df['onset_time'])
    output_df['time_to_mri'] = pd.to_datetime(output_df['mri_date']) - pd.to_datetime(output_df['onset_time'])
    output_df['time_to_iat'] = pd.to_datetime(output_df['iat_start']) - pd.to_datetime(output_df['onset_time'])
    output_df['time_to_ivt'] = pd.to_datetime(output_df['ivt_start']) - pd.to_datetime(output_df['onset_time'])
    output_df['ct_to_iat'] = pd.to_datetime(output_df['iat_start']) - pd.to_datetime(output_df['ct_date'])
    output_df['ct_to_mri'] = pd.to_datetime(output_df['mri_date']) - pd.to_datetime(output_df['ct_date'])

    output_df = output_df[
        ['patient_identifier', 'combined_id', 'combined_id_match', 'Nom', 'Prénom', 'birth_date', 'anonymised_id', 'id_hospital_case',
         'age', 'sex', 'event_type',
         'onset_time', 'ct_date', 'time_to_ct', 'mri_date', 'time_to_mri', 'ct_to_mri',
         'treat_ivt', 'ivt_start', 'time_to_ivt', 'treat_iat', 'iat_start', 'time_to_iat', 'ct_to_iat',
         'NIH admission', 'NIH_24h', 'TICI'
         ]]
    duplicates_df = output_df[output_df.duplicated(subset='combined_id', keep=False)]
    output_df = output_df.drop_duplicates(subset='combined_id')

    print('Sex:\n', output_df['sex'].value_counts())
    print('Median age', np.median(output_df['age'].astype(float)), 'std', np.std(output_df['age'].astype(float)), 'min/max',
          np.min(output_df['age'].astype(float)), np.max(output_df['age'].astype(float)))
    print('Median time to CT', np.median(output_df['time_to_ct']).astype('timedelta64[m]'),
          'min/max', np.min(output_df['time_to_ct']).seconds / 60.0, np.max(output_df['time_to_ct']).seconds / 60.0)
    print('Median time to MRI', np.median(output_df['time_to_mri']).astype('timedelta64[h]'),
          'min/max', np.min(output_df['time_to_mri']).seconds / 3600.0,
          np.max(output_df['time_to_mri']).seconds / 3600.0)
    print('Median NIHSS at admission', np.median(output_df['NIH admission']), 'min/max',
          np.min(output_df['NIH admission']), np.max(output_df['NIH admission']))
    print('Median NIHSS at 24h', np.median(output_df['NIH_24h']), 'min/max',
          np.min(output_df['NIH_24h']), np.max(output_df['NIH_24h']))
    print('Cardinal iat:\n', output_df['treat_iat'].value_counts())
    print('Cardinal ivt:\n', output_df['treat_ivt'].value_counts())

    # print(np.min(output_df['ct_to_mri']))
    # print(output_df['time_to_ct'].head())

    duplicates_df.to_excel(os.path.join(os.path.dirname(patient_info_path), 'duplicates_df.xlsx'))
    output_df.to_excel(os.path.join(os.path.dirname(patient_info_path), 'output_df.xlsx'))

extract_patient_characteristics(
    '/Users/julian/temp/anon_key_2017.xlsx',
    '/Users/julian/temp/190419_Données 2015-16-17.xlsx', id_sheet = 'Sheet1', info_sheet = 'Sheet1 (2)')