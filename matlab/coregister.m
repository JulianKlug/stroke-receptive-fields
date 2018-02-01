%% Perfusion Co-registration script
% This script co-registers perfusions maps to skull-stripped native CT sequences.
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = 'E:\MASTER\realigned_data';
% Subject folders
subjects = {
'Barlovic_Radojka_19480907'
};


sequences = {
    'RAPID_CBF'
    };


% sequences = {
%     'RAPID_CBF' ,...
%     'RAPID_CBV', ...
%     'RAPID_MTT_[s]', ...
%     'RAPID_TMax_[s]'
%     };

realignment_prefix = 'ral_';

% Base image to realign to
base_image_dir = 'E:\MASTER\betted_data';
base_image_name = 'betted';
base_image_ext = '.nii.gz';

%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )
    % Sequences to be realigned (folders)
%     sequences = dir(fullfile(data_path,subjects{i}));
%     sequences = {sequences(24:end).name};

    base_image = fullfile(base_image_dir, subjects{i}, strcat(base_image_name, '.nii'));
    if (~ exist(base_image))
        zipped_base_image = fullfile(base_image_dir, subjects{i}, strcat(base_image_name, base_image_ext));
        gunzip(zipped_base_image);
    end
        
    % load realigned data for each sequence without a prompt
    for j = 1: numel(sequences)
        cd(fullfile(data_path, subjects{i}, sequences{j}));
        accepted_files = dir(strcat(realignment_prefix, '*.nii'));
        input = fullfile(data_path, subjects{i}, ...
            sequences{j}, accepted_files(1).name);
        % display which subject and sequence is being processed
        fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
            subjects{i}, char(sequences), sprintf('%d',size (input ,1)));
        %% SAVE AND RUN JOB
        %
        coregistration = coregister_job(base_image, input);
        log_file = fullfile(fullfile(data_path, subjects{i}, sequences{j}), 'coregister.mat');
        save(log_file, 'coregistration');
        spm('defaults', 'FMRI');
        spm_jobman('run', coregistration);
    end  
        
end

