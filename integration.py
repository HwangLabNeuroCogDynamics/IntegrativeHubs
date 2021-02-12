import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to
import nilearn
import scipy
import os
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nilearn import masking
from nilearn import plotting
from nilearn.image import resample_to_img

def test_script():
    print('test is a test of the IntegrationHub script')


# organize 7T data into nii format
thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
MNI_template = nib.load('/data/backed_up/shared/standard/fsl_mni152/MNI152_T1_2mm.nii')
morel = nib.load('/home/kahwang/RDSS/tmp/Morel.nii.gz')
#morel = resample_to_img(morel, MNI_template, interpolation='nearest')
m = morel.get_fdata()
new_m = m.copy()
new_m[new_m!=0] = 0
#morel = nilearn.image.new_img_like(morel, m)
#morel_vec = masking.apply_mask(morel, morel)


#flat_m = m.reshape((902629), order = 'F')
#flat_m[:] = 0
df = pd.read_csv('/home/kahwang/RDSS/tmp/mac_7T/thal_voxels.csv')
coor_df = pd.read_csv('/home/kahwang/RDSS/tmp/mac_7T/coor.csv', header = None)
#test_image = masking.unmask(df['Thal_ROI'], morel, order='A')
#plotting.plot_roi(test_image)
#plotting.show()

for i in np.arange(0, len(df)):
    voxel = df.loc[i, 'Voxel']

    new_m[coor_df.loc[coor_df[0] == voxel, 1].values[0]-1,  coor_df.loc[coor_df[0] == voxel, 2].values[0]-1, coor_df.loc[coor_df[0] == voxel, 3].values[0]-1] = 1.0*df.loc[i, 'Voxel']

thalamus_mask = nilearn.image.new_img_like(MNI_template, new_m)
thalamus_mask.to_filename('/home/kahwang/RDSS/tmp/mac_7T/Thalmus_vox_idx.nii.gz')

new_m = m.copy()
new_m[new_m!=0] = 0

for i in np.arange(0, len(df)):
    voxel = df.loc[i, 'Voxel']

    new_m[coor_df.loc[coor_df[0] == voxel, 1].values[0]-1,  coor_df.loc[coor_df[0] == voxel, 2].values[0]-1, coor_df.loc[coor_df[0] == voxel, 3].values[0]-1] = 1.0 #*df.loc[i, 'Voxel']

thalamus_mask = nilearn.image.new_img_like(MNI_template, new_m)
thalamus_mask.to_filename('/home/kahwang/RDSS/tmp/mac_7T/Thalmus_vox_idx_mask.nii.gz')

#plotting.plot_roi(thalamus_mask)
#plotting.show()

### End
