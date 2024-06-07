import numpy as np
import pandas as pd
import os
import nibabel as nib
import glob
from datetime import datetime
import scipy
import nilearn
from nilearn.maskers import NiftiLabelsMasker
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp

def write_stats_to_vol_yeo_template_nifti(graph_metric, fn, roisize = 418):
	'''short hand to write vol based nifti file of the stats
	, voxels in each parcel will be replaced with the stats'''


	vol_template = nib.load('/mnt/nfs/lss/lss_kahwang_hpc/ROIs/Schaefer400+Morel+BG_2.5.nii.gz')
	v_data = vol_template.get_fdata()
	graph_data = np.zeros((np.shape(v_data)))

	for i in np.arange(roisize):
		graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn)

### run group level stats and make niis or plots

data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/'
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
roi_fn = "whole_brain"
models = ['color', 'context', 'shape',
       'context:color:shape',
        'response',  'task', 'stim',
        'EDS', 'IDS', 'condition',
        'context:EDS', 'shape:IDS', 'color:IDS']


### check null distribution

nulls = np.zeros((59,418,9,1000)) #sub by roi by var by permutations
for i, s in enumerate(subjects):
    nulls[i,:,:,:] = np.load("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/%s_whole_brain_permutated_stats.npy" %s)

mean_null = np.mean(nulls, axis = (0))

# Convert to DataFrame
df = pd.DataFrame(mean_null[:,4,:].T)  # Transpose to have 1000 rows and 418 columns

# Overlapping histograms with transparency
plt.figure(figsize=(12, 8))
for col in df.columns:
    sns.histplot(df[col], bins=30, kde=False, alpha=0.3)

plt.show()

# plt.figure(figsize=(12, 8))
# for i in range(mean_null.shape[0]):
#     plt.hist(mean_null[i,1, :], bins=30, alpha=0.3, color='blue', edgecolor='black')

# # plt.title('Histograms of 418 Distributions')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# plt.show()

### run stats

results = []
for s in subjects:
    results.append(pd.read_csv(data_dir +"%s_%s_stats.csv" %(s, roi_fn)))

results_df = pd.concat(results, ignore_index=True)

stats = []
for r in results_df.ROI.unique():
    for m in models:
        tdf = results_df.loc[(results_df['ROI']==r) & (results_df['parameter']==m)]
        #model = smf.ols(formula="coef ~ 1", data=tdf).fit()
        t_stat, p_value = ttest_1samp(tdf['coef'], 0)
        ttdf = pd.DataFrame({
            't-statistic': [t_stat],
            'p-value': [p_value],
            'mean': [np.nanmean(tdf['coef'])],
            'ROI': r,
            'model': m
        })
        stats.append(ttdf)    
stats_df = pd.concat(stats, ignore_index=True)
stats_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/group_stats.csv')

### create nii images

for m in models:
    fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic.nii.gz' %m
    write_stats_to_vol_yeo_template_nifti(stats_df.loc[stats_df['model']== m]['t-statistic'].values, fn, roisize = 418)
      
#end