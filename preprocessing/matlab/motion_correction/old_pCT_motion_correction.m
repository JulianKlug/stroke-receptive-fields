%% Perfusion CT Motion correction
% This script applies motion correction to the 4D perfusion CT
% It follows a particular directory structure which must
% be adhered to.
% Prequisites: DPABI package, spm12

% Issues: In some situations the time dependent signal seems to regress to
% mean with this motion correction algorithm

%% Clear variables and command window
clear all , clc
%% Specify paths
% Main folder containing individual subject folders
data_path = '/Users/julian/Desktop/pct_to_reprocess/reworked';

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
                
    try 
        % Check if subject already processed
        if do_not_recalculate && exist(fullfile(CT_folder, strcat('mc_', VPCT_file.name)))
            fprintf('Skipping subject "%s" as normalised files are already present.\n', subjects{i});
            continue;
        end
        % Apply motion correction to this file
        apply_motion_correction(VPCT_file, CT_folder);
    catch ME
        % subjects where motion correction fails are gathered in a file
        fileID = fopen(fullfile(data_path, 'motion_correction_failed.txt'),'at');
        fprintf(fileID,'%s %s %s %s\n', subjects{i}, ME.stack(1).name, ME.stack(1).line, ME.message);
        fclose(fileID);
    end
end

