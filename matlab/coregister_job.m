% matlabbatch = coregister_job(reference, source)
% The function takes a reference file and a source file and passes them to
% matlabbatch for coregistration
% them on to the matlabbatch sturcture array

function matlabbatch = coregister_job(reference, source)
% 
% matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'E:\MASTER\betted_data\Barlovic_Radojka_19480907\betted.nii,1'};
% matlabbatch{1}.spm.spatial.coreg.estwrite.source = {'E:\MASTER\realigned_data\Barlovic_Radojka_19480907\RAPID_CBF\rRAPID_CBF_Perfusion_20160103102300_431.nii,1'};
matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {reference};
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {source};
matlabbatch{1}.spm.spatial.coreg.estwrite.other = {''};
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'coreg_';
