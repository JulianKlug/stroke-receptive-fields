%% Perfusion Normalisation wrapper script
% This script normalises perfusions maps to skull-stripped native MNI-CT 
% by using the native CT that the perfusion maps must have been coregistered to.
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '/Volumes/stroke_hdd1/stroke_db/2016/temp/extracted_test';

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
template_dir = '/Users/julian/master/stroke-predict/preprocessing/matlab/normalisation';
if(use_stripped_template)
    ct_template = fullfile(template_dir, 'scct_stripped.nii');
else
    ct_template = fullfile(template_dir, 'scct.nii');
end

sequences = {
    'CBF' ,...
    'CBV', ...
    'MTT', ...
    'Tmax'
    };

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
    modalities = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    modality = modalities(1).name;

    base_image = fullfile(base_image_dir, subjects{i}, modality, ...
        strcat(base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
    if (~ exist(base_image))
        zipped_base_image = strcat(base_image, '.gz');
        gunzip(zipped_base_image);
    end
        
    % load coregistered data for each sequence without a prompt
    input = {};
    for j = 1: numel(sequences)
        input{end + 1} = fullfile(data_path, subjects{i}, modality, ...
                            strcat('coreg_', sequences{j}, '_' ,subjects{i}, '.nii'));

    end
    
   
    %% RUN NORMALISATION
    
   % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
        subjects{i}, strjoin(sequences), sprintf('%d',size (input ,2)));
    
    base_image_to_warp = fullfile(base_image_dir, subjects{i}, modality, ...
    strcat('reor_', base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
    copyfile(base_image, base_image_to_warp);

%         Script using modern SPM12 normalise function
    normalise_to_CT(base_image_to_warp, input, ct_template);

    
%% (SHOULD NOT BE USED ANYMORE) Script based on Clinical_CT toolbox based on SPM8 normalise 
%           --> not successful
%     lesionMask = '';
%     mask = 1;
%     bb = [-78 -112 -50
%         78 76 85];
%     vox = [1 1 1];
%     DelIntermediate = 0;
%     AutoSetOrigin = 1;
%     useStrippedTemplate = use_stripped_template;
%     perf_clinical_ctnorm(base_image_to_warp, lesionMask, input, vox, bb,DelIntermediate, mask, useStrippedTemplate, AutoSetOrigin);

        
end

