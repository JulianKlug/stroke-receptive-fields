# stroke-predict
This repository hosts scripts for a research project on prediction of ischemic stroke.

## Pipeline

Extraction Verification
- verification/find_empty_folders.py : find empty folders in subject directories hinting towards failed exports and save in excel file
- verification/verify_RAPID37.py : check that all subjects with perfusion CTs have 37 RAPID images (and not 11)
- utils/extract_unknown_studies_folder.py : extract images saved as an unspecified "study" folder

Pre :

- 1. organise.py : organise into a new working directory, extracting only useful and renaming to something sensible
- 2. to_nii_batch.py : batch convert subject DICOMs to Nifti
- 3. flatten.py : flatten into an MRI and a CT folder
- 4. verify_completeness.py : verify that all necessary files are present

CT :

- skull_strip/skull_strip_wrapper.py : batch skull strip native CTs of multiple patients
- matlab/perfCT_coregistration_wrapper.m : coregister perfusion CT to native CT
- matlab/perfCT_normalisation_wrapper.m : normalise perfusion CT and native CT to CT_MNI

MRI :

- matlab/mri_coreg_normalisation_wrapper.m : recenter subject CT, co-register T2 to subject CT, co-register T2 to CT-MNI and normalise to CT-MNI

Post:
As RAPID performs excessive skull-stripping, same crop has to be applied to lesion maps to remove lesions without underlying input data.
- masking/brain_mask.py : create brain masks based on RAPID perfusion maps
- masking/mask_lesions.py : apply brain masks to lesions

## Requirements

- matlab
- spm12 : http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
- Clinical Toolbox for SPM (https://www.nitrc.org/projects/clinicaltbx/) [Has to be in the folder spm12/toolbox]
- dcm2niix : https://github.com/rordenlab/dcm2niix
