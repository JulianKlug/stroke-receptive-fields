%% Perfusion Realignment script
% This script realigns perfusions maps to skull-stripped native CT sequences.
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = 'E:\MASTER\realigned_data';
out_path = 'E:\MASTER\realigned_data';
% Subject folders
subjects = {
'Barlovic_Radojka_19480907'
};

sequences = {
    'RAPID_CBF' ,...
    'RAPID_CBV', ...
    'RAPID_MTT_[s]', ...
    'RAPID_TMax_[s]'
    };

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
    
    outdir = cellstr ( fullfile (out_path , subjects{i}));
%     if (~ exist(outdir{1}))
%         mkdir(outdir{1});
%     end
%     
    input = {{base_image}};
    
    % load raw data for each sequence without a prompt
    for j = 1: numel(sequences)
        cd(fullfile(data_path, subjects{i}, sequences{j}));
        for t = 1:numel(dir('*.nii'))
            files = dir('*.nii')
            input{1}{end + 1} = fullfile(data_path, subjects{i}, ...
                sequences{j}, files(t).name);
        end
    end  
        
    % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
        subjects{i}, char(sequences), sprintf('%d',size (input ,1)));
    % set up SPM - readable structure array
    input{1} = input{1}.';
    realignment = realign_job(input , outdir);
    %% SAVE AND RUN JOB
    %
    log_file = fullfile(outdir, 'realign.mat');
    save(log_file{1}, 'realignment');
    spm('defaults', 'FMRI');
    spm_jobman('run', realignment);

end
