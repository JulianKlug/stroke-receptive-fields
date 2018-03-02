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
data_path = '/Users/julian/master/data/preprocessing_test';
% Subject folders
subjects = {
'Barlovic_Radojka_19480907'
};

use_stripped_template = false;

sequences = {
    'RAPID_CBF' ,...
    'RAPID_CBV', ...
    'RAPID_MTT_[s]', ...
    'RAPID_TMax_[s]'
    };

% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = '';
if(use_stripped_template)
    base_image_prefix = 'betted_';
end
base_image_ext = '.nii.gz';

%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )
    modalities = dir(fullfile(data_path,subjects{i}, 'Ct2*'));
    modality = modalities(1).name;

    base_image = fullfile(base_image_dir, subjects{i}, modality, ...
        strcat(base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
    if (~ exist(base_image))
        zipped_base_image = strcat(base_image, '.gz');
        gunzip(zipped_base_image);
    end
        
    % load coregistered data for each sequence without a prompt
    for j = 1: numel(sequences)
        input = fullfile(data_path, subjects{i}, modality, ...
                     strcat('coreg_', sequences{j}, '_' ,subjects{i}, '.nii'));


        % display which subject and sequence is being processed
        fprintf('Processing subject "%s" , "%s" (%s files )\n' ,...
            subjects{i}, char(sequences), sprintf('%d',size (input ,1)));
        %% RUN NORMALISATION
        %
        lesionMask = '';
        mask = 1;
        bb = [-78 -112 -50
            78 76 85];
        vox = [1 1 1];
        DelIntermediate = 0;
        AutoSetOrigin = 1;
        useStrippedTemplate = use_stripped_template;
       
        
        base_image_to_warp = fullfile(base_image_dir, subjects{i}, modality, ...
        strcat('w_', base_image_prefix, 'SPC_301mm_Std_', subjects{i}, '.nii'));
        copyfile(base_image, base_image_to_warp);
    
        if(use_stripped_template)
            ct_template = fullfile('/Users/julian/master/stroke-predict/matlab/normalisation/scct_stripped.nii');
        else
            ct_template = fullfile('/Users/julian/master/stroke-predict/matlab/normalisation/scct.nii');            
        end
        
%         Script using modern SPM12 normalise function
        normalise_to_CT(base_image_to_warp, input, ct_template);

%         Script based on Clinical_CT toolbox based on SPM8 normalise 
%           --> not successful
%         perf_clinical_ctnorm(base_image_to_warp, lesionMask, input, vox, bb,DelIntermediate, mask, useStrippedTemplate, AutoSetOrigin);
    end  
        
end

