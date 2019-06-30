# stroke-predict
This repository hosts scripts for a research project on prediction of ischemic stroke.

## Preprocessing Pipeline

Extraction Verification
- pre_verification/find_empty_folders.py : find empty folders in subject directories hinting towards failed exports and save in excel file
- pre_verification/verify_RAPID37.py : check that all subjects with perfusion CTs have 37 RAPID images (and not 11)
- utils/extract_unknown_studies_folder.py : extract images saved as an unspecified "study" folder
- utils/extract_RAPID37_folder.py : extract images saved as an unspecified "RAPID37" folder

Pre :

1. organise.py : organise into a new working directory, extracting only useful and renaming to something sensible + anonymize patient information
    - Nb.: watch out for patient seperator, patient might be missed if they use a different seperator in their file/folder names
2. to_nii_batch.py : batch convert subject DICOMs to Nifti
3. flatten.py : flatten into an MRI and a CT folder
4. verify_completeness.py : verify that all necessary files are present

CT :

0. utils/resolve_RAPID_4D_maps: resolve RAPID maps with 4 dimensions
    - get_RAPID_4D_list: find subjects with 4D RAPID maps
    - resolve_RAPID_4D_maps : reduce dimensions to 3D of given subjects (subjects may need to be downloaded from the server first as this function requires an Xserver for graphical feedback)
1. skull_strip/skull_strip_wrapper.py : batch skull strip native CTs of multiple patients and segment CSF
2. matlab/perfCT_coregistration_wrapper.m : coregister perfusion CT to betted native CT
3. matlab/perfCT_normalisation_wrapper.m : normalise perfusion CT and native CT to CT_MNI

MRI :

- matlab/mri_coreg_normalisation_wrapper.m : recenter subject CT, co-register T2 to subject CT, co-register T2 to CT-MNI and normalise to CT-MNI

Post:
As RAPID performs excessive skull-stripping, same crop has to be applied to lesion maps to remove lesions without underlying input data. 
At the same time the CSF_mask is integrated into the non-brain mask.
- masking/brain_mask.py : create brain masks based on RAPID perfusion maps
- masking/mask_lesions.py : apply brain masks to lesions

## Requirements

- matlab
- spm12 : http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
- Clinical Toolbox for SPM (https://www.nitrc.org/projects/clinicaltbx/) [Has to be in the folder spm12/toolbox]
- dcm2niix : https://github.com/rordenlab/dcm2niix
