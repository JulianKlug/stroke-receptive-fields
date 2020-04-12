%% Perfusion CT Motion correction
% This script applies motion correction to the 4D perfusion CT
% It follows a particular directory structure which must
% be adhered to.
% Prequisites: DPABI package, spm12

%% Clear variables and command window
clear all , clc
%% Specify paths
% Main folder containing individual subject folders
data_path = '/home/klug/data/original_data/with_pct_2016_2017';

% Options
do_not_recalculate = true;

% Prequisite paths
DPABI_path = '/home/klug/utils/DPABI_V4.3_200401';
spm_path = '/home/klug/spm12';
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
    fprintf('%s %i/%i (%i%%) \n', subjects{i}, i, size(subjects, 1), 100 * i / size(subjects, 1));

    modalities = dir(fullfile(data_path,subjects{i}, 'pCT*'));
    if length(modalities) < 1
      fprintf('Skipping "%s" as pCT folder can not be found.\n', subjects{i});
      continue;
    end
    modality = modalities(1).name;
    
    CT_folder = fullfile(data_path, subjects{i}, modality);
    VPCT_file = dir(fullfile(CT_folder, ...
                    strcat('VPCT_', subjects{i}, '.nii*')));
    % Check if subject already processed
    if do_not_recalculate && exist(fullfile(CT_folder, strcat('mc_', VPCT_file.name)))
        fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
        continue;
    end
    % Apply motion correction to this file
    apply_motion_correction(VPCT_file, CT_folder);
end
