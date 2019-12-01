%% Perfusion Co-registration script
% This script co-registers perfusions maps to skull-stripped native CT sequences.
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '/Users/julian/temp/extraction_bv40/trial1';
spm_path = '/Users/julian/Documents/MATLAB/spm12';
do_not_recalculate = true; 

script_path = mfilename('fullpath');
script_folder = script_path(1 : end - size(mfilename, 2));
addpath(genpath(script_folder));
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

sequences = {
    'CBF' ,...
    'CBV', ...
    'MTT', ...
    'Tmax'
    };

% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = 'betted';
base_image_ext = '.nii.gz';

%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )
    fprintf('%i/%i (%i%%) \n', i, size(subjects, 1), 100 * i / size(subjects, 1));
    modalities = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    modality = modalities(1).name;

% Verify if coreg was already done
    coreg_count = 0;
    for jj = 1: numel(sequences)
        coreg_sequences = dir(fullfile(base_image_dir, subjects{i}, modality, ...
            strcat('coreg_', sequences{jj}, '_', subjects{i}, '*', '.nii*')));
        try
            if exist(fullfile(base_image_dir, subjects{i}, modality, coreg_sequences(1).name))
                coreg_count = coreg_count + 1;
            end
        catch ME
        end
    end
    if coreg_count == size(sequences, 2) && do_not_recalculate
        fprintf('Skipping subject "%s" as coregistered files are already present.\n', subjects{i});
        continue;
    end

    base_image_list = dir(fullfile(base_image_dir, subjects{i}, modality, ...
        strcat(base_image_prefix, '_SPC_301mm_Std_', subjects{i}, '*', '.nii*')));
    base_image = fullfile(base_image_dir, subjects{i}, modality, base_image_list(1).name);
    [filepath,name,ext] = fileparts(base_image);
    if strcmp(ext, '.gz') 
        gunzip(base_image);
        base_image = base_image(1: end - 3);
    end
        
    % load realigned data for each sequence without a prompt
    for j = 1: numel(sequences)
%         cd(fullfile(data_path, subjects{i}, modality));
%         accepted_files = dir(strcat(realignment_prefix, '*.nii'));
        input = fullfile(data_path, subjects{i}, modality, ...
            strcat(sequences{j}, '_' ,subjects{i}, '.nii'));
        % display which subject and sequence is being processed
        fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
            subjects{i}, char(sequences{j}), sprintf('%d',size (input ,1)));
        %% SAVE AND RUN JOB
        %
        coregistration = coregister_job(base_image, input, {});
        log_file = fullfile(data_path, subjects{i}, modality, ...
            'logs',strcat(sequences{j}, '_coreg.mat'));
        mkdir(fullfile(data_path, subjects{i}, modality, 'logs'));
        save(log_file, 'coregistration');
        spm('defaults', 'FMRI');
        spm_jobman('run', coregistration);
    end  
        
end

