import os
import numpy as np
import pandas as pd
from . import input_parameters
from . import final_parameters

INPUT_PARAMETERS = input_parameters.parameters
input_schema = input_parameters.validation_schema

FINAL_PARAMETERS = final_parameters.parameters
final_schema = final_parameters.validation_schema

def load_clinical_data(ids, data_dir, filename, sheet = 'Sheet1'):
    """
    Load clinical data from excel Sheet
        - Clean Input data
        - Compute needed variables
    What data is needed and it's requirements are specified in the files :
        - input_parameters : what parameters should be used as input
        - final_parameters : what parameters should be given as output

    Args:
        ids : list of patient ids for which the data should be extracted
        data_dir : directory containing the excel Sheet
        filename : name of the file (excel)
        sheet (optional) : name of the sheet from which to extract the data from

    Returns:
        'clinical_data': numpy array containing the data for each of the patients [patient, (n_parameters)]
    """
    clinical_data = []
    included_subjects = [] # ie no exclusion criteria for every patient
    filepath = os.path.join(data_dir, filename)

    # Load spreadsheet
    xl = pd.ExcelFile(filepath)
    # Load a sheet into a DataFrame
    df = xl.parse(sheet)

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

    print(len(FINAL_PARAMETERS), 'clinical variables found.')
    print(FINAL_PARAMETERS)
    # Final data extraction and validation
    extra_columns = ['id_hospital_case', 'treat_iat_before_ct'] # other columns needed for subject evaluation
    final_parameter_data = cleaned_parameter_data.filter(items=FINAL_PARAMETERS + extra_columns)
    final_errors = final_schema.validate(final_parameter_data.filter(items=FINAL_PARAMETERS)) # do no validate ID

    # Check if there are negative values
    if (sum(n < 0 for n in final_parameter_data.values.flatten()) > 0):
        raise ValueError('Negative values found in initial data. Can not replace NaNs.')
    # As NaNs are not accepted by external memomry XGB, code NaN as -1
    final_parameter_data = final_parameter_data.fillna(-1)

    if (len(final_errors) == 1):
        print(final_errors[0])

    for error in final_errors:
        if not pd.isnull(error.value):
            print(error)
            print('"{}" failed!'.format(str(error.value)))

    for id in ids:
        # Get single subject
        included = True;
        subject_all_data = final_parameter_data[final_parameter_data['id_hospital_case'] == int(id)]

        # exclude patients who received IaT befor imaging
        if (subject_all_data['treat_iat_before_ct'].values[0] == 1):
            included = False
            print('IaT performed before imaging. Patient excluded.', id)

        try:
            subject_clinical_data = subject_all_data.drop(columns = extra_columns).values[0]
            pass
        except IndexError as e:
            included = False
            subject_clinical_data = np.NaN
            print('No clinical data found for this subject:', id)

        included_subjects.append(included)
        clinical_data.append(subject_clinical_data)

    return (np.array(included_subjects), np.array(clinical_data))
