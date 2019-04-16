%% MRI Pre-processing wrapper script
% This script preprocesses T2 MRI images and without the associated lesion maps 
% 1. Coregister T2 to native CT (non betted, with reset origin if needed)
% 2. Coregister native CT to CT-MNI and apply same transformation  to the coregistered T2 (output from step 1)
% 3. Normalise coregistered native CT (at step 2)  to CT-MNI and apply same transformation to the twice-coregistered T2 (output from step 2)
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
addpath(genpath(pwd))
%% Specify paths
% Experiment folder
data_path = 'C:\Users\Julian\Documents\temp\anon_dir';
spm_path = 'C:\Users\Julian\Documents\MATLAB\spm12';
do_not_recalculate = true; 
addpath(genpath(spm_path));

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

use_stripped_template = false; % normalisation works better with skull
template_dir = fullfile(pwd, '/normalisation');
if(use_stripped_template)
    ct_template = fullfile(template_dir, 'scct_stripped.nii');
else
    ct_template = fullfile(template_dir, 'scct.nii');
end

% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = '';
if(use_stripped_template)
    base_image_prefix = 'betted_';
end
base_image_ext = '.nii.gz';

addpath(template_dir, data_path)
%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )
    fprintf('%i/%i (%i%%) \n', i, size(subjects, 1), 100 * i / size(subjects, 1));
    ct_dir = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    ct_dir = ct_dir.name;

    mri_dir = dir(fullfile(data_path,subjects{i}, 'MRI*'));
    if (~ isempty(mri_dir))
        mri_dir = mri_dir.name;
    else
        mri_dir = dir(fullfile(data_path,subjects{i}, 'Irm*'))
        mri_dir = mri_dir.name;
    end
    
    wcoreg_sequences = dir(fullfile(data_path,subjects{i}, mri_dir, ...
            strcat('wcoreg_', 't2_tse_tra', '_', subjects{i}, '*', '.nii*')));
    try
        if exist(fullfile(data_path, subjects{i}, mri_dir, wcoreg_sequences(1).name))... 
                && do_not_recalculate
            fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
            continue;
        end
    catch ME
    end
        
%   base_image is the native CT (bettet or not betted, depending on prefix)
    original_base_image_list = dir(fullfile(base_image_dir, subjects{i}, ct_dir, ...
        strcat(base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '*', '.nii*')));
    original_base_image = fullfile(base_image_dir, subjects{i}, ct_dir, original_base_image_list(1).name);
    [filepath,name,ext] = fileparts(original_base_image);
    if strcmp(ext, '.gz') 
        gunzip(original_base_image);
        original_base_image = original_base_image(1: end - 3);
    end

    base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
    copyfile(original_base_image, base_image);
    
    % Create a new image that will be recentered with setOrigin() $
    % (but not co-registered)
    center_base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat('c_',base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
    copyfile(original_base_image, center_base_image);
    setOrigin(center_base_image, false, 3);
        
    % load data for each sequence without a prompt
    mri_files =  dir(fullfile(data_path, subjects{i}, mri_dir, strcat('t2_tse_tra_', subjects{i}, '.nii')));
    mri_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 mri_files.name);

    % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" \n' ,...
        subjects{i}, mri_files.name);
    
    %% COREGISTRATION to native CT
    
    coregistration_to_native = coregister_job(center_base_image, mri_input, {});
    log_file = fullfile(data_path, subjects{i}, mri_dir, ...
        'logs',strcat('to_SPC_301mm_Std_', 'coreg.mat'));
    mkdir(fullfile(data_path, subjects{i}, mri_dir, 'logs'));
    save(log_file, 'coregistration_to_native');
    spm('defaults', 'FMRI');
    spm_jobman('run', coregistration_to_native);
    
    %% COREGISTRATION to CT-MNI
    % Co-register native CT (recentered) to CT MNI and apply
    % transformation to MRI 
    
    coreg_mri_input = fullfile(data_path, subjects{i}, mri_dir, ...
                                   strcat('coreg_', mri_files.name));
                        
    % Use coregistration and resetting origin 
    % (ref: rorden lab Clinical toolbox)
    setOrigin(strvcat(center_base_image, coreg_mri_input), true, 3);
                               

    %% RUN NORMALISATION
    %
    
    if ~exist(ct_template) %file does not exist
        vols = spm_select(inf,'image','Select template to co-register and normalise to');
    end
    
    
    % Normalisation script
    % based on Rorden Lab's clinical toolbox
    % adapted for SPM12 and perfusion CT
    normalise_to_CT(center_base_image, {coreg_mri_input}, ct_template);
        
end

