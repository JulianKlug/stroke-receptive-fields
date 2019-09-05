%% DWI MRI Pre-processing wrapper script
% This script coregisters DWI to T2 images
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '/Users/julian/temp/nifti_extracted copy';

if ~(exist(data_path))
    fprintf('Data directory does not exist. Please enter a valid directory.')
end

spm_path = '/Users/julian/Documents/MATLAB/spm12';
script_path = mfilename('fullpath');
script_folder = script_path(1 : end - size(mfilename, 2));
addpath(genpath(script_folder));
addpath(genpath(spm_path));

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

    mri_dir = dir(fullfile(data_path,subjects{i}, 'MRI*'));
    if (~ isempty(mri_dir))
        mri_dir = mri_dir.name;
    else
        mri_dir = dir(fullfile(data_path,subjects{i}, 'Irm*'))
        mri_dir = mri_dir.name;
    end
    
    coreg_sequences = dir(fullfile(data_path, subjects{i}, mri_dir, ...
            strcat('wcoreg_', 't2_tse_tra', '_', subjects{i}, '*', '.nii*')));
    wcoreg_VOI = fullfile(data_path, subjects{i}, mri_dir, ...
                            strcat('wcoreg_','VOI_', subjects{i}, '.nii'));
    wcoreg_ADC = fullfile(data_path, subjects{i}, mri_dir, ...
                            strcat('wcoreg_','ADC_', subjects{i}, '.nii'));
    wcoreg_TRACE = fullfile(data_path, subjects{i}, mri_dir, ...
                            strcat('wcoreg_','TRACE_', subjects{i}, '.nii'));
    try
        if exist(fullfile(data_path,subjects{i}, mri_dir, wcoreg_sequences(1).name))...
                && ((with_VOI && exist(wcoreg_VOI)) || ~with_VOI) ...
                && ((with_DWI && exist(wcoreg_ADC) && exist(wcoreg_TRACE)) || ~with_DWI) ...
                && do_not_recalculate
            fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
            continue;
        end
    catch ME
    end
    
%   base_image is the T2
    base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 't2_tse_tra_', subjects{i}, '.nii'));

  
    % load data for each sequence without a prompt               
    ADC_sequence = fullfile(data_path, subjects{i}, mri_dir, ...
                   strcat('ADC_', subjects{i}, '.nii'));
    TRACE_sequence = fullfile(data_path, subjects{i}, mri_dir, ...
                   strcat('TRACE_', subjects{i}, '.nii')); 
            
    % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" \n' ,...
        subjects{i}, ADC_sequence);
    
    %% COREGISTRATION to native T2
    coregistration_to_t2 = coregister_job(base_image, ADC_sequence, {TRACE_sequence});
    coregistration_to_t2{1,1}.spm.spatial.coreg.estwrite.roptions.prefix = 't2_';

    log_file = fullfile(data_path, subjects{i}, mri_dir, ...
        'logs',strcat('to_T2_', 'coreg.mat'));
    mkdir(fullfile(data_path, subjects{i}, mri_dir, 'logs'));
    save(log_file, 'coregistration_to_t2');
    spm('defaults', 'FMRI');
    spm_jobman('run', coregistration_to_t2);

end
