%% Perfusion CT Motion correction
% This script applies motion correction to the 4D perfusion CT
% It follows a particular directory structure which must
% be adhered to.
% Prequisites: DPABI package, spm12

%% Clear variables and command window
clear all , clc
%% Specify paths
% Main folder containing individual subject folders
data_path = '/Users/julian/temp/VPCT_extraction_test/pipeline_test_compa';

% Options
do_not_recalculate = true;

% Prequisite paths
DPABI_path = '/Users/julian/Documents/MATLAB/DPABI_V4.3_200301';
spm_path = '/Users/julian/Documents/MATLAB/spm12';
script_path = mfilename('fullpath');
script_folder = script_path(1 : end - size(mfilename, 2));
addpath(genpath(script_folder));
addpath(genpath(spm_path));
addpath(genpath(DPABI_path));
addpath(data_path)

% Select subjects based on folders in data_path
d = dir(data_path);
isub = [d(:).isdir]; %# returns logical vector
subjects = {d(isub).name}';
subjects(ismember(subjects,{'.','..'})) = [];

% loop through subjects to process
for i = 1: numel(subjects)
    fprintf('%i/%i (%i%%) \n', i, size(subjects, 1), 100 * i / size(subjects, 1));

    modalities = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    modality = modalities(1).name;
    
    CT_folder = fullfile(data_path, subjects{i}, modality);
    VPCT_file = dir(fullfile(CT_folder, ...
                    strcat('VPCT_', subjects{i}, '*', '.nii*')));
                
    % Check if subject already processed
    if do_not_recalculate && exist(fullfile(CT_folder, strcat('mc_', VPCT_file.name)))
        fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
        continue;
    end
    
    % Apply motion correction to this file
    apply_motion_correction(VPCT_file, CT_folder);
end