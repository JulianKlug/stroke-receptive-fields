import os
import numpy as np
import pandas as pd
import input_parameters
import final_parameters

data_dir = '/Users/julian/OneDrive - unige.ch/master project/clinical_data'
filename = 'orig_cleaned_clinical_data_2016.xlsx'
filepath = os.path.join(data_dir, filename)

# Load spreadsheet
xl = pd.ExcelFile(filepath)

# Load a sheet into a DataFrame by name: df1
df = xl.parse('Sheet1')

INPUT_PARAMETERS = input_parameters.parameters
input_schema = input_parameters.validation_schema

FINAL_PARAMETERS = final_parameters.parameters
final_schema = final_parameters.validation_schema

# DATA CLEANING
parameter_data = df.filter(items=INPUT_PARAMETERS + ['id_hospital_case'])
cleaned_parameter_data = parameter_data.replace('?', np.NaN)

df_obj = cleaned_parameter_data.select_dtypes(['object'])
# remove trailing spaces and cast to lower case
cleaned_parameter_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip()).apply(lambda x: x.str.lower())
# replace white spaces by underscores
cleaned_parameter_data['treat_ivt'] = cleaned_parameter_data['treat_ivt'].apply(lambda x: x.replace(' ', '_'))
cleaned_parameter_data['onset_known'] = cleaned_parameter_data['onset_known'].apply(lambda x: x.replace(' ', '_'))

input_errors = input_schema.validate(cleaned_parameter_data.filter(items=INPUT_PARAMETERS)) # do no validate ID

if (len(input_errors) == 1):
    print(input_errors[0])

for error in input_errors:
    if not pd.isnull(error.value):
        print(error)
        print('"{}" failed!'.format(error.value))

# Variable computation and binarization
# treat_ivt_before_ct should be 1 if ivt was started before the ct, and 0 otherwise
cleaned_parameter_data['treat_ivt_before_ct'] = np.where((cleaned_parameter_data['ivt_start'] < cleaned_parameter_data['Firstimage_date']) | (cleaned_parameter_data['treat_ivt'] == 'started_before_admission'), 1, 0)

# treat_iat_before_ct should be 1 if iat was started before the ct, and 0 otherwise
cleaned_parameter_data['treat_iat_before_ct'] = np.where(cleaned_parameter_data['iat_start'] < cleaned_parameter_data['Firstimage_date'], 1, 0)

# Calculate time from onset to CT in minutes
cleaned_parameter_data['onset_to_ct'] = (cleaned_parameter_data['Firstimage_date'] - cleaned_parameter_data['onset_time']).astype('timedelta64[m]')

# dummy encode this onset_known
onset_known_dummies = pd.get_dummies(cleaned_parameter_data['onset_known'])
onset_known_dummies = onset_known_dummies.rename(columns={'no': 'onset_known_no', 'yes': 'onset_known_yes', 'wake_up': 'onset_known_wake_up'})
cleaned_parameter_data = pd.concat([cleaned_parameter_data, onset_known_dummies], axis = 1)

# binarization of binary variables
cleaned_parameter_data['sex'] = np.where(cleaned_parameter_data['sex'] == 'm', 1, 0)
cleaned_parameter_data['hypertension'] = np.where(cleaned_parameter_data['hypertension'] == 'yes', 1, 0)
cleaned_parameter_data['diabetes'] = np.where(cleaned_parameter_data['diabetes'] == 'yes', 1, 0)
cleaned_parameter_data['hyperlipidemia'] = np.where(cleaned_parameter_data['hyperlipidemia'] == 'yes', 1, 0)
cleaned_parameter_data['smoking'] = np.where(cleaned_parameter_data['smoking'] == 'yes', 1, 0)
cleaned_parameter_data['atrialfib'] = np.where(cleaned_parameter_data['atrialfib'] == 'yes', 1, 0)
cleaned_parameter_data['stroke_pre'] = np.where(cleaned_parameter_data['stroke_pre'] == 'yes', 1, 0)
cleaned_parameter_data['tia_pre'] = np.where(cleaned_parameter_data['tia_pre'] == 'yes', 1, 0)
cleaned_parameter_data['ich_pre'] = np.where(cleaned_parameter_data['ich_pre'] == 'yes', 1, 0)
cleaned_parameter_data['treat_antipatelet'] = np.where(cleaned_parameter_data['treat_antipatelet'] == 'yes', 1, 0)
cleaned_parameter_data['treat_anticoagulant'] = np.where(cleaned_parameter_data['treat_anticoagulant'] == 'yes', 1, 0)

# Final data extraction and validation
final_parameter_data = cleaned_parameter_data.filter(items=FINAL_PARAMETERS + ['id_hospital_case'])
final_errors = final_schema.validate(final_parameter_data.filter(items=FINAL_PARAMETERS)) # do no validate ID

if (len(final_errors) == 1):
    print(final_errors[0])

for error in final_errors:
    if not pd.isnull(error.value):
        print(error)
        print('"{}" failed!'.format(str(error.value)))

# Get single subject
subject_all_data = final_parameter_data[final_parameter_data['id_hospital_case'] == 97664863]
subject_clinical_data = subject_all_data.drop(columns=['id_hospital_case']).values[0]

print(len(subject_clinical_data))
# print(subject_all_data.ix[:, 'Firstimage_date'])
# print(subject_all_data.ix[:, 'iat_start'])

# print(subject_all_data.ix[:, 'treat_iat_before_ct'])

# print(subject_all_data.ix[:, 'onset_known_no'], subject_all_data.ix[:, 'onset_known_yes'], subject_all_data.ix[:, 'onset_known_wake_up'])


# print(df[df['id_hospital_case'] == 898729].iloc[0]['sex'])
# print(subset.ix[:, 'sex'].index)
