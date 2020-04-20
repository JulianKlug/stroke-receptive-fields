import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm

spm_path = '/home/klug/spm12'
mlab.MatlabCommand.set_default_paths(spm_path)

print(spm.SPMCommand().version)  
