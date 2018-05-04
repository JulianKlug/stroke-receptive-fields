# stroke-predict
This repository hosts scripts for a research project on prediction of ischemic stroke.

## Pipeline

Pre :

- to_nii_batch.py : batch convert subject DICOMs to Nifti
- organise.py : organise into a new working directory, extracting only useful and renaming to something sensible

CT :

- skull_strip/skull_strip_wrapper.py : batch skull strip native CTs of multiple patients
- matlab/perfCT_coregistration_wrapper.m : coregister perfusion CT to native CT
- matlab/perfCT_normalisation_wrapper.m : normalise perfusion CT and native CT to CT_MNI

MRI :

- matlab/mri_coreg_normalisation_wrapper.m : recenter subject CT, co-register T2 to subject CT, co-register T2 to CT-MNI and normalise to CT-MNI

## Requirements

- matlab
- spm12 : http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
- Clinical Toolbox for SPM (https://www.nitrc.org/projects/clinicaltbx/) [Has to be in the folder spm12/toolbox]
- dcm2niix : https://github.com/rordenlab/dcm2niix
