%% Perfusion Co-registration script
% This script co-registers perfusions maps to skull-stripped native CT sequences.
% It follows a particular directory structure which must
% be adhered to.


%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '/Users/julian/temp/VPCT_extraction_test/pipeline_test';
spm_path = '/Users/julian/Documents/MATLAB/spm12';
do_not_recalculate = true; 
% with_pCT = true;

script_path = mfilename('fullpath');
script_folder = script_path(1 : end - size(mfilename, 2));
addpath(genpath(script_folder));
addpath(genpath(spm_path));

if ~(exist(data_path))
    fprintf('Data directory does not exist. Please enter a valid directory.')
end


% Select subjects based on folders in data_path
d = dir(data_path);
isub = [d(:).isdir]; 
subjects = {d(isub).name}';
subjects(ismember(subjects,{'.','..'})) = [];

sequences = {
    'CBF' ,...
    'CBV', ...
    'MTT', ...
    'Tmax'
    };

% perfusion_ct_name = 'mc_VPCT'; % VPCT has ben motion corrected before

% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = 'betted';
base_image_ext = '.nii.gz';

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
    else

    %% Coregister perfusion maps to SPC
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
            input = fullfile(data_path, subjects{i}, modality, ...
                strcat(sequences{j}, '_' ,subjects{i}, '.nii'));
            % display which subject and sequence is being processed
            fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
                subjects{i}, char(sequences{j}), sprintf('%d',size (input ,1)));
            
            % SAVE AND RUN JOB
            coregistration = coregister_job(base_image, input, {});
            log_file = fullfile(data_path, subjects{i}, modality, ...
                'logs',strcat(sequences{j}, '_coreg.mat'));
            mkdir(fullfile(data_path, subjects{i}, modality, 'logs'));
            save(log_file, 'coregistration');
            spm('defaults', 'FMRI');
            spm_jobman('run', coregistration);
        end
    end
    
    %% Coregister 4D pCT to SPC
    % Run 4D pCT coregistration seperately as it is not aligned with the
    % other images
%     if with_pCT
%         if do_not_recalculate && length(dir(fullfile(base_image_dir, subjects{i}, ...
%                  modality, '4D_split', strcat('coreg_', perfusion_ct_name, '*')))) > 1
%              fprintf('Coregistration for 4D perfusion CT for subject "%s" already done.\n', subjects{i});
%              continue;
%         end
% 
%         fprintf('Processing 4D perfusion CT for subject "%s"\n', subjects{i});
%         pCT_file = fullfile(data_path, subjects{i}, modality, ...
%             strcat(perfusion_ct_name, '_' ,subjects{i}, '.nii'));
%         non_betted_base_image = fullfile(base_image_dir, subjects{i}, modality, ...
%         strcat('SPC_301mm_Std_', subjects{i},'.nii'));
%         
%         % Split 4D file in many 3D files as SPM can not handle 4D files
%         split_4D_subfolder = fullfile(data_path, subjects{i}, modality, '/4D_split');
%         mkdir(split_4D_subfolder);
%         spm_file_split(pCT_file, split_4D_subfolder);
% 
%         split_4D_file = textscan(ls(fullfile(split_4D_subfolder, '*.nii')), '%s', 'delimiter', '\n');
%         first_3D_subimage = split_4D_file{1}{1};
%         remaining_3D_subimages = split_4D_file{1}(2:end);
%         
%         %% SAVE AND RUN JOB
%         coregistration = coregister_job(non_betted_base_image, first_3D_subimage, remaining_3D_subimages);
%         log_file = fullfile(data_path, subjects{i}, modality, ...
%             'logs',strcat(perfusion_ct_name, '_coreg.mat'));
%         mkdir(fullfile(data_path, subjects{i}, modality, 'logs'));
%         save(log_file, 'coregistration');
%         spm('defaults', 'FMRI');
%         spm_jobman('run', coregistration);
%     end
        
end

