import os
import numpy as np
import pandas as pd
from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, InListValidation

data_dir = '/Users/julian/OneDrive - unige.ch/master project/clinical_data'
filename = 'mod_clinical_data_2016.xlsx'
filepath = os.path.join(data_dir, filename)

# Load spreadsheet
xl = pd.ExcelFile(filepath)

# Load a sheet into a DataFrame by name: df1
df = xl.parse('Sheet1')

PARAMETERS = [
    'age',
    'sex',
    'height',
    'weight',
    'BMI',
    'onset_known',
    'NIH admission',
    'bp_syst',
    'bp_diast',
    'glucose',
    'créatinine',
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
    'treat_anticoagulant',
    'treat_ivt',
    'treat_iat',
]

schema = Schema([
    Column('age', [InRangeValidation(1, 120)]),
    Column('sex', [InListValidation(['m', 'f'])]),
    Column('height', [InRangeValidation(50, 300)]),
    Column('weight', [InRangeValidation(10, 400)]),
    Column('BMI', [InRangeValidation(10, 60)]),

    Column('onset_known', [InListValidation(['yes', 'no', 'wake_up'])]),
    Column('NIH admission', [InRangeValidation(0, 43)]),
    Column('bp_syst', [InRangeValidation(0, 300)]),
    Column('bp_diast', [InRangeValidation(0, 300)]),
    Column('glucose', [InRangeValidation(0.1, 30)]),
    Column('créatinine', [InRangeValidation(0.1, 1000)]),

    Column('hypertension', [InListValidation(['yes', 'no'])]),
    Column('diabetes', [InListValidation(['yes', 'no'])]),
    Column('hyperlipidemia', [InListValidation(['yes', 'no'])]),
    Column('smoking', [InListValidation(['yes', 'no'])]),
    Column('atrialfib', [InListValidation(['yes', 'no'])]),

    Column('stroke_pre', [InListValidation(['yes', 'no'])]),
    Column('tia_pre', [InListValidation(['yes', 'no'])]),
    Column('ich_pre', [InListValidation(['yes', 'no'])]),

    Column('treat_antipatelet', [InListValidation(['yes', 'no'])]),
    Column('treat_anticoagulant', [InListValidation(['yes', 'no'])]),
    Column('treat_ivt', [InListValidation(['yes', 'no', 'started_before_admission'])]),
    Column('treat_iat', [InListValidation(['yes', 'no'])]),
])

# DATA CLEANING
parameter_data = df.filter(items=PARAMETERS)
cleaned_parameter_data = parameter_data.replace('?', np.NaN)
df_obj = cleaned_parameter_data.select_dtypes(['object'])
# remove trailing spaces and cast to lower case
cleaned_parameter_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip()).apply(lambda x: x.str.lower())
# replace white spaces by underscores
cleaned_parameter_data['treat_ivt'] = cleaned_parameter_data['treat_ivt'].apply(lambda x: x.replace(' ', '_'))
cleaned_parameter_data['onset_known'] = cleaned_parameter_data['onset_known'].apply(lambda x: x.replace(' ', '_'))
# zero BMI should not exist
cleaned_parameter_data[cleaned_parameter_data['BMI'] == 0] = np.NaN

errors = schema.validate(cleaned_parameter_data)

for error in errors:
    if not pd.isnull(error.value):
        print(error)
        print('"{}" failed!'.format(error.value))

# for param in parameters:
#     print(param, df[df['id_hospital_case'] == 898729][param])

subject_all_data = df[df['id_hospital_case'] == 898729]
subject_clinical_data = list(map(lambda parameter: subject_all_data.iloc[0][parameter], PARAMETERS))

# print(subject_all_data)
# print(subject_clinical_data)

# print(df[df['id_hospital_case'] == 898729].iloc[0]['sex'])
# print(subset.ix[:, 'sex'].index)
