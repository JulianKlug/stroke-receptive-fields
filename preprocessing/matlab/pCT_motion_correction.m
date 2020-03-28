main_dir = '/Users/julian/temp/VPCT_extraction_test/spm_motion_corr/realigned_split';
DPABI_path = '/Users/julian/Documents/MATLAB/DPABI_V4.3_200301';

addpath(genpath(DPABI_path));

%% Load the initial 4D data
V0_file = dir(fullfile(main_dir, 'VPCT*.nii'));
V0_header = spm_vol(fullfile(main_dir, V0_file(1).name));
V0 = spm_read_vols(V0_header);

mkdir(fullfile(V0_file(1).folder, '/4D_split'))
spm_file_split(fullfile(main_dir, V0_file(1).name), fullfile(V0_file(1).folder, '/4D_split'));
%% build X
%1. add constant linear & quadratic trends
% CSFavg = ones(512*512*37,1);
CSFavg = ones(size(V0,4), 1);
X=[ones(numel(CSFavg),1) [1:numel(CSFavg)]'/numel(CSFavg) ...
    [1:numel(CSFavg)].^2'/(numel(CSFavg)^2)];
%2. add motion params:
mot = dir(fullfile(main_dir,'rp*.txt')); %%(Because I called the file rp... just set it differently in case)
Cov=load(fullfile(main_dir,mot(1).name));
X=[X,Cov(:,1:6)]; %take first 6 (not derivative if they are there)
 
V0Cov=zeros(size(V0));
V0idx=(1:size(V0,4));
for i=1:size(V0,1)
    for j=1:size(V0,2)
        for k=1:size(V0,3)
            [beta,res,SSE,SSR,T] = y_regress_ss(squeeze(V0(i,j,k,V0idx)),X); %GLM
            V0Cov(i,j,k,V0idx)=res+mean(squeeze(V0(i,j,k,V0idx))); %put the mean signal back
        end
    end
end
V0Cov(isnan(V0Cov))=0;
V0Cov_header = V0_header(1);
[V0Cov_header.fname] = deal('outputfile.nii');
rest_Write4DNIfTI(V0Cov, V0Cov_header, 'outputfile.nii')
