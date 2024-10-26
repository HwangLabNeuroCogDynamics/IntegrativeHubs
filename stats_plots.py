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
from scipy.stats import ttest_rel
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


t_nulls = np.zeros((418,len(models),4096))
# for r in range(418):
#     for v in range(len(models)):
#         t_nulls[r, v, :], _ = ttest_1samp(nulls[:, r, v, :], 0, axis=0, nan_policy='omit')

# Use Parallel and delayed to parallelize the nested loops
from joblib import Parallel, delayed
def perform_ttest(r, v):
    return ttest_1samp(nulls[:, r, v, :], 0, axis=0, nan_policy='omit')[0]

results = Parallel(n_jobs=-1)(delayed(perform_ttest)(r, v) for r in range(418) for v in range(len(models)))

# Reshape the results and assign to t_nulls
for idx, (r, v) in enumerate([(r, v) for r in range(418) for v in range(len(models))]):
    t_nulls[r, v, :] = results[idx]


np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/t_nulls.npy", t_nulls)

# quick plot to take a look at null distribution
# df = pd.DataFrame(t_nulls[:,4,:].T) 
# plt.figure(figsize=(12, 8))
# for col in np.arange(415,418):
#     sns.histplot(df[col], bins=30, kde=False, alpha=0.3)
# plt.show()
## remarkbly, the 97.5 percentile of the null seems to be around t=2, very similar to parametric test. But with some diff between ROIs, so perhaps ROI specific null is a good idea.

### run stats using empirical p
permutation_stats = []
t_nulls = np.load("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/t_nulls.npy")

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
                'p-value': 1-(np.mean(t_stat>t_nulls[r,i,:].flatten())),
                'mean': [np.nanmean(tdf['coef'])],
                'ROI': r+1,
                'model': m
            })
            permutation_stats.append(ttdf)    
permutation_stats_df = pd.concat(permutation_stats, ignore_index=True)
permutation_stats_df['q'] = fdrcorrection(permutation_stats_df['p-value'])[1]
permutation_stats_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/permutation_stats_df.csv')


# create thresholded nii images

permutation_stats_df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/permutation_stats_df.csv")       
thres_niis = {}
for m in models:
    if m != 'Intercept':
        fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic_thresholded.nii.gz' %m
        metric = permutation_stats_df.loc[permutation_stats_df['model']== m]['t-statistic'].values
        mask = permutation_stats_df.loc[permutation_stats_df['model']== m]['q'].values < .05
        #mask[np.argsort(metric)[:-15]]=0
        thres_niis[m] = write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 400)
        
### create nii images
# for m in models:
#     fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic.nii.gz' %m
#     metric = permutation_stats_df.loc[permutation_stats_df['model']== m]['t-statistic'].values
#     mask = 1
#     write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 418)

plotting.plot_glass_brain(thres_niis['context'], display_mode='lzry', plot_abs=False,
                          title='', threshold=0.1)
plt.show()

plotting.plot_glass_brain(thres_niis['feature_shape'], display_mode='lzry', plot_abs=False,
                          title='', threshold=0.1)
plt.show()

plotting.plot_glass_brain(thres_niis['feature_color'], display_mode='lzry', plot_abs=False,
                          title='', threshold=0.1)
plt.show()

plotting.plot_glass_brain(thres_niis['task'], display_mode='lzry', plot_abs=False,
                        title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['response'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()


permutation_stats_df.loc[permutation_stats_df['ROI']==405]

#compare to af results
af_df = pd.read_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/400ROIs_voxels/af/59subs_ncaf_max.csv')
af_rsa = {}
for m in models:
    if m != 'Intercept':
        #fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_t-statistic_thresholded.nii.gz' %m
        metric = permutation_stats_df.loc[permutation_stats_df['model']== m]['t-statistic'].values
        mask = permutation_stats_df.loc[permutation_stats_df['model']== m]['q'].values < .05
        #roi_idx = np.where(mask)[0]+1
        roi_idx = np.intersect1d(np.argsort(metric)[-15:], np.where(mask)) + 1
        #roi_idx = np.argsort(metric)[-15:]+1
        adf = af_df.loc[af_df['roi'].isin(roi_idx)]
        adf = adf.groupby('subjects').mean()
        af_rsa[m] = adf['All_GLT'].values
        # print(m)
        # print("EDSvIDS: ", ttest_rel(adf['EDS'], adf['IDS']))
        # print("IDSvstay: ", ttest_rel(adf['IDS'], adf['Stay']))

print("context v feature: ", ttest_rel(af_rsa['context'], af_rsa['context:color']))
print("context v color: ", ttest_rel(af_rsa['context'], af_rsa['context:shape']))
#print("context v shape: ", ttest_rel(af_rsa['context'], af_rsa['task']))
# print("context v task: ", ttest_rel(af_rsa['context'], af_rsa['task']))
# print("context v response: ", ttest_rel(af_rsa['context'], af_rsa['response']))



# plot evoked responses
EDS_v_IDS = nib.load("/home/kahwang/argon/data/ThalHi/3dMEMA/cluster_corrected_mema_EDS-IDS_59subs_tvalue.nii.gz")
plotting.plot_glass_brain(EDS_v_IDS, display_mode='lzry', plot_abs=False,
                          title='EDS - IDS', threshold=2)
plt.show()



######################
### run parametric stats on switch effects
data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/'
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
roi_fn = "whole_brain"

results = []
for s in subjects:
    results.append(pd.read_csv(data_dir +"%s_%s_switch_stats.csv" %(s, roi_fn)))
results_df = pd.concat(results, ignore_index=True)
models = results_df.parameter.unique()

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
#stats_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/group_stats.csv')

# create thresholded nii images      
thres_niis = {}
for m in models:
    if m != 'Intercept':
        fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_switch_t-statistic_thresholded.nii.gz' %m
        metric = stats_df.loc[stats_df['model']== m]['t-statistic'].values
        mask = stats_df.loc[stats_df['model']== m]['q'].values < .05
        mask[np.argsort(abs(metric))[:-3]]=0
        thres_niis[m] = write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 400)
        
plotting.plot_glass_brain(thres_niis['context'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['EDS:context'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['rt:EDS'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()


plotting.plot_glass_brain(thres_niis['rt'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['error'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()

plotting.plot_glass_brain(thres_niis['response'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()


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

# ## plot PC
# pc_img = nib.load("/home/kahwang/bin/IntegrativeHubs/data/pc_vectors.nii.gz")
# plotting.plot_glass_brain(pc_img, display_mode='lzry', plot_abs=False,
#                           title='Network Hub', threshold=0.7)
# plt.show()


# ## plot AF
# af_df = pd.read_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/Activity_Flow/noise_ceiling/corticocortical/af/59subs_ncaf_max.csv')
# af_vec = af_df.groupby('roi').mean()['All_GLT'].values
# fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/af.nii.gz'
# af_nii = write_stats_to_vol_yeo_template_nifti(af_vec, fn, roisize = 400)
# plotting.plot_glass_brain(af_nii, display_mode='lzry', plot_abs=False,
#                           title='Activity Flow', threshold=0.63)
# plt.show()


#       t-statistic   p-value      mean  ROI          model         q
# 2400     1.527353  0.065761  0.000337  401        context  0.163458
# 2401     0.206256  0.412981  0.000025  401           task  0.569651
# 2402     0.166267  0.428449  0.000017  401       response  0.583044
# 2403     0.977686  0.164586  0.000280  401  context:color  0.304411
# 2404     0.515502  0.299857  0.000142  401  context:shape  0.449786
# 2405     1.624831  0.054565  0.000374  401       identity  0.144509


#end