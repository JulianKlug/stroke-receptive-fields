from unidecode import unidecode
import hashlib
import pandas as pd
import os

def anonymise_subject(clinical_row):
    if not isinstance(clinical_row['Nom'], str):
        return clinical_row
    last_name = unidecode(clinical_row['Nom'].strip().upper())
    first_name = unidecode(clinical_row['Prénom'].strip().upper())
    patient_birth_date = str(clinical_row['birth_date'].year) \
                         + str(clinical_row['birth_date'].month).zfill(2) \
                         + str(clinical_row['birth_date'].day).zfill(2)

    subject_key = last_name + '^' + first_name + '^' + patient_birth_date

    ID = hashlib.sha256(subject_key.encode('utf-8')).hexdigest()[:8]
    pid = 'subj-' + str(ID)

    clinical_row['subject_key'] = subject_key
    clinical_row['pid'] = pid
    return clinical_row

def anonymise_clinical_data(clinical_data_path, sheet_name='Sheet1', save_anonymised=False):
    clinical_df = pd.read_excel(clinical_data_path, sheet_name=sheet_name)
    clinical_df = clinical_df.apply(anonymise_subject, axis=1)

    if save_anonymised:
        anonymised_name = 'anonymised_' + os.path.basename(clinical_data_path)
        clinical_df.to_excel(os.path.join(os.path.dirname(clinical_data_path), anonymised_name))

    return clinical_df

anonymise_clinical_data('/Users/julian/temp/190419_Données 2015-16-17.xlsx', 'Sheet1 (2)', save_anonymised=True)

