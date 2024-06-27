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
from nilearn import plotting
from nilearn import datasets
from nilearn import surface
from statsmodels.stats.multitest import fdrcorrection
plt.ion()

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
	return new_nii

### load subject level output
data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/'
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
roi_fn = "whole_brain"

results = []
for s in subjects:
    results.append(pd.read_csv(data_dir +"%s_%s_stats.csv" %(s, roi_fn)))
results_df = pd.concat(results, ignore_index=True)
models = results_df.parameter.unique()

### check null distribution
nulls = np.zeros((59,418,len(models),4096)) #sub by roi by var by permutations
for i, s in enumerate(subjects):
    nulls[i,:,:,:] = np.load("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/%s_whole_brain_permutated_stats.npy" %s)

# Convert to DataFrame
mean_null = np.mean(nulls, axis = (0))
df = pd.DataFrame(mean_null[:,4,:].T)  # Transpose to have 1000 rows and 418 columns

# Overlapping histograms with transparency
plt.figure(figsize=(12, 8))
for col in df.columns:
    sns.histplot(df[col], bins=30, kde=False, alpha=0.3)
plt.show()

t_nulls = np.zeros((418,len(models),4096))
for p in np.arange(4096):
     for r in np.arange(418):
          for v in np.arange(len(models)):
               t_nulls[r,v,p], _= ttest_1samp(nulls[:,r,v,p], 0, nan_policy ='omit')


df = pd.DataFrame(t_nulls[:,9,:].T) 
plt.figure(figsize=(12, 8))
for col in np.arange(415,418):
    sns.histplot(df[col], bins=30, kde=False, alpha=0.3)
plt.show()
## remarkbly, the 97.5 percentile of the null seems to be around t=2. But with some diff between ROIs, so perhaps ROI specific null is a good idea.

### run stats using empirical p
permutation_stats = []
for r in range(418): #results_df.ROI.unique()
    for i, m in enumerate(models):
        if i == 0: #no intercept
            continue
        else:
            tdf = results_df.loc[(results_df['ROI']==r+1) & (results_df['parameter']==m)]
            #model = smf.ols(formula="coef ~ 1", data=tdf).fit()
            t_stat, _ = ttest_1samp(tdf['coef'], 0)
            
            ttdf = pd.DataFrame({
                't-statistic': [t_stat],
                'p-value': 1-(np.mean(t_stat>t_nulls[r,1:,:].flatten())),
                'mean': [np.nanmean(tdf['coef'])],
                'ROI': r+1,
                'model': m
            })
            permutation_stats.append(ttdf)    
permutation_stats_df = pd.concat(permutation_stats, ignore_index=True)
permutation_stats_df['q'] = fdrcorrection(permutation_stats_df['p-value'])[1]
permutation_stats_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/permutation_stats_df.csv')


# create thresholded nii images      
thres_niis = {}
for m in models:
    if m != 'Intercept':
        fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic_thresholded.nii.gz' %m
        metric = permutation_stats_df.loc[permutation_stats_df['model']== m]['t-statistic'].values
        mask = permutation_stats_df.loc[permutation_stats_df['model']== m]['q'].values < .05
        mask[np.argsort(metric)[:-10]]=0
        thres_niis[m] = write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 400)

### create nii images
# for m in models:
#     fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic.nii.gz' %m
#     metric = permutation_stats_df.loc[permutation_stats_df['model']== m]['t-statistic'].values
#     mask = 1
#     write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 418)

plotting.plot_glass_brain(thres_niis['context'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['shape'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['color'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['feature'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['task'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['response'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['stim'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

permutation_stats_df.loc[permutation_stats_df['ROI']==405]


# plot evoked responses
EDS_v_IDS = nib.load("/home/kahwang/argon/data/ThalHi/3dMEMA/cluster_corrected_mema_EDS-IDS_59subs_tvalue.nii.gz")
plotting.plot_glass_brain(EDS_v_IDS, display_mode='lzry', plot_abs=False,
                          title='EDS - IDS', threshold=2)
plt.show()


### run parametric stats
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

# FDR correction
stats_df['q'] = fdrcorrection(stats_df['p-value'])[1]
stats_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/group_stats.csv')

### use nilearn plotting to visualize results
fsaverage = datasets.fetch_surf_fsaverage()
curv_right = surface.load_surf_data(fsaverage.curv_right)
curv_right_sign = np.sign(curv_right)
texture = surface.vol_to_surf(thres_niis['shape'], fsaverage.pial_right)

fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture, hemi='right',
    title='Surface right hemisphere', colorbar=True,
    threshold=0.1, bg_map=curv_right_sign,
)
fig.show()

## plot PC
pc_img = nib.load("/home/kahwang/bin/IntegrativeHubs/data/pc_vectors.nii.gz")
plotting.plot_glass_brain(pc_img, display_mode='lzry', plot_abs=False,
                          title='Network Hub', threshold=0.7)
plt.show()


## plot AF
af_df = pd.read_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/corticocortical/af/59subs_ncaf_max.csv')
af_vec = af_df.groupby('roi').mean()['All_GLT'].values
fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/af.nii.gz'
af_nii = write_stats_to_vol_yeo_template_nifti(af_vec, fn, roisize = 400)
plotting.plot_glass_brain(af_nii, display_mode='lzry', plot_abs=False,
                          title='Activity Flow', threshold=0.63)
plt.show()

#end