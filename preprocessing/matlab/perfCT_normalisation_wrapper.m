%% Perfusion Normalisation wrapper script
% This script normalises perfusions maps and CSF-mask to native MNI-CT (better if not skull-stripped) 
% by using the non-betted native CT that the perfusion maps must have been coregistered to.
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = '//Users/julian/temp/extraction_kv90/newBET_kv90';
spm_path = '/Users/julian/Documents/MATLAB/spm12';
do_not_recalculate = false; 
with_angio = true;

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

use_stripped_template = false; % normalisation works better with skull
template_dir = fullfile(script_folder, '/normalisation');
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

angio_ct_name = 'extracted_betted_Angio_CT_075_Qr40_3_A_90kV';
angio_ct_suffix = '.nii.gz_Mask';
% angio_ct_name = 'extracted_betted_Angio_CT_075_Bv40';
% angio_ct_suffix = '';
csf_mask_name = 'CSF_mask.nii';

% Base image to co-register to
base_image_dir = data_path;
base_image_name = 'SPC_301mm_Std_';
base_image_prefix = '';
betted_prefix = 'betted_';
if(use_stripped_template)
    base_image_prefix = 'betted_';
end
base_image_ext = '.nii.gz';

addpath(template_dir, data_path)
%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )
    fprintf('%i/%i (%i%%) \n', i, size(subjects, 1), 100 * i / size(subjects, 1));
    modalities = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    modality = modalities(1).name;

% Verify if normalisation was already done
    wcoreg_count = 0;
    for jj = 1: numel(sequences)
        coreg_sequences = dir(fullfile(base_image_dir, subjects{i}, modality, ...
            strcat('wcoreg_', sequences{jj}, '_', subjects{i}, '*', '.nii*')));
        csf_mask_names = dir(fullfile(base_image_dir, subjects{i}, modality, ...
            strcat('wcoreg_', csf_mask_name,'*')));
        try
            if exist(fullfile(base_image_dir, subjects{i}, modality, coreg_sequences(1).name)) | ...
                exist(fullfile(base_image_dir, subjects{i}, modality, csf_mask_names(1).name))
                wcoreg_count = wcoreg_count + 1;
            end
        catch ME
        end
    end
    if wcoreg_count == size(sequences, 2) + 1 && do_not_recalculate
        fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
        continue;
    end    
    
    base_image_list = dir(fullfile(base_image_dir, subjects{i}, modality, ...
        strcat(base_image_prefix, base_image_name, subjects{i}, '*', '.nii*')));
    base_image = fullfile(base_image_dir, subjects{i}, modality, base_image_list(1).name);
    [filepath,name,ext] = fileparts(base_image);
    if strcmp(ext, '.gz') 
        gunzip(base_image);
        base_image = base_image(1: end - 3);
    end

    % load coregistered data for each sequence without a prompt
    input = {};
    for j = 1: numel(sequences)
        input{end + 1} = fullfile(data_path, subjects{i}, modality, ...
                            strcat('coreg_', sequences{j}, '_' ,subjects{i}, '.nii'));
    end
    
    if with_angio
       angio_file_list = dir(fullfile(data_path, subjects{i}, modality, ...
                            strcat(angio_ct_name, '_' ,subjects{i}, angio_ct_suffix, '.nii*')));
       angio_file = fullfile(data_path, subjects{i}, modality, angio_file_list(1).name);
       [filepath,name,ext] = fileparts(angio_file);
        if strcmp(ext, '.gz') 
            gunzip(angio_file);
            angio_file = angio_file(1: end - 3);
        end
        input{end + 1} = angio_file
    end 

    % Coregister CSF mask of betted image to template as well
    csf_mask_list = dir(fullfile(base_image_dir, subjects{i}, modality, ...
        strcat(csf_mask_name, '*')));
    csf_mask_image = fullfile(base_image_dir, subjects{i}, modality, csf_mask_list(1).name);
    [filepath,name,ext] = fileparts(csf_mask_image);
    if strcmp(ext, '.gz') 
        gunzip(csf_mask_image);
        csf_mask_image = csf_mask_image(1: end - 3);
    end
    input{end + 1} = csf_mask_image;
   
    %% RUN NORMALISATION
    
   % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" + "%s" (%s files)\n' ,...
        subjects{i}, strjoin(sequences), csf_mask_name, sprintf('%d',size (input ,2)));
    
    base_image_to_warp = fullfile(base_image_dir, subjects{i}, modality, ...
    strcat('reor_', base_image_prefix, base_image_name, subjects{i}, '.nii'));
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

