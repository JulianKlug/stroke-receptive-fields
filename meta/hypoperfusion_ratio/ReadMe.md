# Perfusion Volumes

## Tmax Thresholds 

For Tmax thesholded volumes, a simple threshold is applied above:

- 6 sec.: defines penumbra
- 10 sec.: even more hypoperfused tissue

## Definition of ischemic volume

To define the ischemic core, CBF is first normalised to the median value in:
 1. The contralateral side  
 2. Regions with Tmax < 4 sec. 
 
 Then a threshold is applied if the normalized CBF is < 30%
 
 ## Smoothing 
 
Smoothing is attempted by sequentially applying:
1. binary closing with a structuring element width a 3D-ball element with a 3 voxel diameter 
2. binary opening with a structuring element width a 3D-ball element with a 3 voxel diameter 
3. removing all elements smaller than 1000 voxels with a type 1 connectivity