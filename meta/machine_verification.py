import pydicom
import os
import pandas as pd

main_dir = '/Users/julian/master/data'
data_dir = os.path.join(main_dir, 'working_data')
output_dir = os.path.join(main_dir, 'meta')
ct_sequences = ['SPC_301mm_Std', 'VPCT_Perfusion_4D_50_Hr36']
mri_sequences = ['t2_tse_tra', 'T2W_TSE_tra']
sequences = ct_sequences + mri_sequences

columns = ['subject','study', 'Manufacturer', 'ManufacturerModelName', 'Modality', 'MagneticFieldStrength']
machine_data = pd.DataFrame(columns=columns)

mr_machines = []
ct_machines = []

subjects = os.listdir(data_dir)

for subject in subjects:
    subject_dir = os.path.join(data_dir, subject)
    if os.path.isdir(subject_dir):
        modalities = [o for o in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir,o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)

            studies = [o for o in os.listdir(modality_dir)
                            if os.path.isdir(os.path.join(modality_dir,o))]

            for study in studies:
                study_dir = os.path.join(modality_dir, study)
                if study in sequences:
                    dcms = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
                    dcm = pydicom.dcmread(os.path.join(study_dir, dcms[0]))
                    magnetic_field = 0
                    if (dcm.Modality == 'MR'):
                        magnetic_field = dcm.MagneticFieldStrength
                    temp_df = pd.DataFrame([[subject, study, dcm.Manufacturer,
                        dcm.ManufacturerModelName, dcm.Modality, magnetic_field]],
                        columns = columns)
                    machine_data = machine_data.append(temp_df)

                    if (dcm.Modality == 'MR'):
                        if not (dcm.ManufacturerModelName in mr_machines):
                            mr_machines.append(dcm.ManufacturerModelName)
                    else:
                        if not (dcm.ManufacturerModelName in ct_machines):
                            ct_machines.append(dcm.ManufacturerModelName)

unique_mrs = pd.DataFrame({'MR':mr_machines})
unique_cts = pd.DataFrame({'CT': ct_machines})

with pd.ExcelWriter(os.path.join(output_dir, 'machine_info.xlsx')) as writer:
    machine_data.to_excel(writer, sheet_name='all_info')
    unique_mrs.to_excel(writer, sheet_name='MRI_models')
    unique_cts.to_excel(writer, sheet_name='CT_models')
