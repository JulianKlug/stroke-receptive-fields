%% DWI MRI Pre-processing wrapper script
% This script preprocesses T2 MRI images and associated lesion maps 
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '/Users/julian/master/data/DWI_test';

if ~(exist(data_path))
    fprintf('Data directory does not exist. Please enter a valid directory.')
end

% Subject folders

% Select individual subjects
% subjects = {
% 'patient1'
% };

% Or select subjects based on folders in data_path
d = dir(data_path);
isub = [d(:).isdir]; %# returns logical vector
subjects = {d(isub).name}';
subjects(ismember(subjects,{'.','..'})) = [];


% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = '';
age_ext = '.nii.gz';

addpath(data_path)
%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )

    mri_dir = dir(fullfile(data_path,subjects{i}, 'Neuro*'));
    if (~ isempty(mri_dir))
        mri_dir = mri_dir.name;
    else
        mri_dir = dir(fullfile(data_path,subjects{i}, 'Irm*'))
        mri_dir = mri_dir.name;
    end
    
%   base_image is the T2
    base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 'T2W_TSE_tra_', subjects{i}, '.nii'));
    if (~ exist(base_image))
        base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 't2_tse_tra_', subjects{i}, '.nii'));;
    end
%     if (~ exist(original_base_image))
%         zipped_base_image = strcat(original_base_image, '.gz');
%         gunzip(zipped_base_image);
%     end
  
    % load data for each sequence without a prompt
    dwi_files =  dir(fullfile(data_path, subjects{i}, mri_dir,'*ADC*'));
    dwi_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 dwi_files.name);

    lesion_map_initial = fullfile(data_path, subjects{i}, ...
                 strcat('VOI_lesion_', subjects{i}, '.nii'));
    lesion_map = fullfile(data_path, subjects{i}, mri_dir, ...
                 strcat('VOI_lesion_', subjects{i}, '.nii'));     
         
    if (exist(lesion_map_initial))
        movefile(lesion_map_initial, lesion_map);
    end

    % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" \n' ,...
        subjects{i}, dwi_files.name);
    
    %% COREGISTRATION to native T2
    coregistration_to_t2 = coregister_job(base_image, dwi_input, {});
    log_file = fullfile(data_path, subjects{i}, mri_dir, ...
        'logs',strcat('to_T2_', 'coreg.mat'));
    mkdir(fullfile(data_path, subjects{i}, mri_dir, 'logs'));
    save(log_file, 'coregistration_to_t2');
    spm('defaults', 'FMRI');
    spm_jobman('run', coregistration_to_t2);

end

