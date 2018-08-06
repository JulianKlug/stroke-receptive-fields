import os
import pandas as pd

data_dir = '/Users/julian/OneDrive - unige.ch/master project/clinical_data'
filename = 'mod_clinical_data_2016.xlsx'
filepath = os.path.join(data_dir, filename)

# Load spreadsheet
xl = pd.ExcelFile(filepath)

# Load a sheet into a DataFrame by name: df1
df = xl.parse('Sheet1')

parameters = [
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

# for param in parameters:
#     print(param, df[df['id_hospital_case'] == 898729][param])

index = df[df['id_hospital_case'] == 898729].index
print(index)

print(df.ix[index, 'sex'])
# print(subset.ix[:, 'sex'].index)


# 	Treatment
# 	Age at time of stroke
# 	Gender
# 	Onset known / wakeup
# 	NIHSS
# 	Tension
# 	Height weight BMI
# 	Gluc + Creat
# 	IV treatment : y/n
# 	Arterial treatment:  y/n
# 	ATCD stroke
# 	Atcd Hemorragie
# 	FR cardio
