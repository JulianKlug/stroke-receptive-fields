import pandas as pd
import os

default_parameters = [
    'age',
    'sex',
    'height',
    'weight',
    # timing parameters
    'onset_known',
    # parameters needed to calculate onset to ct
    'onset_time',
    'Firstimage_date',
    # clinical parameters
    'NIH admission',
    'bp_syst',
    'bp_diast',
    'glucose', # to discuss
    'créatinine', # to discuss
    # cardiovascular risk factors
    'hypertension',
    'diabetes',
    'hyperlipidemia',
    'smoking',
    'atrialfib',
    # ATCDs
    'stroke_pre',
    'tia_pre',
    'ich_pre',
    # Treatment
    'treat_antipatelet',
    'treat_anticoagulant'
]

def select_clinical_parameters(clinical_data_path, sheet_name='Sheet1', parameters=default_parameters, save_selected=True):
    clinical_df = pd.read_excel(clinical_data_path, sheet_name=sheet_name)
    selected_df = clinical_df.filter(items=['pid'] + parameters)

    if save_selected:
        selected_name = 'selected_' + os.path.basename(clinical_data_path)
        selected_df.to_excel(os.path.join(os.path.dirname(clinical_data_path), selected_name))

    return selected_df

select_clinical_parameters('/Users/julian/temp/anonymised_190419_Données 2015-16-17.xlsx')


