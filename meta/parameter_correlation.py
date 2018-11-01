import sys, os
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import analysis.data_loader as data_loader

data_dir = '/Users/julian/master/data/clinical_data_test'
save = True
CLIN, IN, OUT = data_loader.load_saved_data(data_dir)

Tmax = IN[:,:,:,:,0]
CBF = IN[:,:,:,:,1]
MTT = IN[:,:,:,:,2]
CBV = IN[:,:,:,:,3]

if save:
    plt.ioff()
    plt.switch_backend('agg')

# OVERALL CORRELATION
d = {'Tmax': Tmax.reshape(-1), 'CBF': CBF.reshape(-1), 'MTT': MTT.reshape(-1), 'CBV': CBV.reshape(-1), 'Lesion': OUT.reshape(-1)}
data = pd.DataFrame(data = d)
corr = data.corr(method='pearson')
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)

if save:
    corr.to_csv(os.path.join(data_dir, 'parameter_correlation.csv'))
    plt.savefig(os.path.join(data_dir, 'parameter_correlation.png'))
    plt.close()
else:
    plt.show()

# VOXELWISE CORRELATION
n_subjs, n_x, n_y, n_z = Tmax.shape
n_voxels = n_x * n_y * n_z
vox_Tmax = Tmax.reshape(n_voxels, n_subjs)
vox_MTT = MTT.reshape(n_voxels, n_subjs)
vox_CBV = CBV.reshape(n_voxels, n_subjs)
vox_CBF = CBF.reshape(n_voxels, n_subjs)
vox_Lesion = OUT.reshape(n_voxels, n_subjs)

vox_correlations = np.empty((n_voxels, 5, 5))

for i in range(int(n_voxels)):
    vox_d = {'Tmax': vox_Tmax[i].reshape(-1), 'CBF': vox_CBF[i].reshape(-1), 'MTT': vox_MTT[i].reshape(-1), 'CBV': vox_CBV[i].reshape(-1), 'Lesion': vox_Lesion[i].reshape(-1)}

    vox_data = pd.DataFrame(data = vox_d)
    vox_corr = vox_data.corr(method='pearson')
    vox_correlations[i] = vox_corr.values

vxl_wise_corr = np.nanmean(vox_correlations, axis = 0)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(vxl_wise_corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)

if save:
    df = pd.DataFrame(data = vxl_wise_corr, columns = vox_data.columns)
    df.to_csv(os.path.join(data_dir, 'vxl_wise_parameter_correlation.csv'))
    plt.savefig(os.path.join(data_dir, 'vxl_wise_parameter_correlation.png'))
    plt.close()
else:
    plt.show()
