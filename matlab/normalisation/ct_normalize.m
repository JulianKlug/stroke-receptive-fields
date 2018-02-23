function ct_normalize(main_image, image_to_conormalize, ct_template)

%report if templates are not found
if (exist(ct_template) == 0) %report if files do not exist
    %   fprintf('Please put the CT template in the SPM template folder\n');
    ct_template = spm_select(inf,'image','Select Template to normalize');
    return
end;

if nargin <1 %no files
    main_image = spm_select(inf,'image','Select CT[s] to normalize');
end;

if nargin < 2
    image_to_conormalize = '';
end

% convert to cormack units
% c_main_image = h2cUnits(main_image);
% c_image_to_conormalize = h2cUnits(image_to_conormalize);
c_main_image = main_image;
c_image_to_conormalize = image_to_conormalize;

% set origin
images = strvcat(deblank(c_main_image), deblank(c_image_to_conormalize));
clinical_setorigin(images,3); %coregister to CT

recentered_base_image = images(1,:);
recentered_image_to_conormalize= images(2,:);

matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {strcat(c_main_image, ',1')}; % {'/Users/julian/master/data/test/spm/SPC_301mm_Std_Barlovic_Radojka_19480907.nii,1'};
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {strcat(c_image_to_conormalize, ',1')};

% OPTIONS FROM CLINICAL_CT TOOLBOX
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.template = {strcat(ct_template ,',1')};
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.smosrc = 8;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.smoref = 0;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.regtype = 'mni';
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.cutoff = 25;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.nits = 16;
matlabbatch{1}.spm.spatial.normalise.estwrite.roptions.preserve = 0;
matlabbatch{1}.spm.spatial.normalise.estwrite.roptions.wrap = [0 0 0];

matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/Users/julian/Documents/MATLAB/spm12/tpm/TPM.nii'};
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = 1; % from
% clinical ct
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                             78 76 85]; %[-78 -112 -50; 78 76 85]
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
% matlabbatch{1}.spm.spatial.normalise.estwrite.roptions.interp = 1; from
% clinical ct
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';


spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch);


end