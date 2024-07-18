# code to test if trail evoke response correlate with neural distance (effort for switching)
import numpy as np
import pandas as pd
from scipy.stats import zscore
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn 
import scipy.linalg as linalg

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

#load behavioral data
data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/coefs/'
nii_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/"

df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']

#s = 10001

for s in subjects:
    #load neural correlation distance
    data = np.load(data_dir+ "%s_whole_brain_coef.npy" %s)  # this is in ROI by trial by trial

    num_trials = data.shape[2]
    roi_num = data.shape[0]
    run_breaks = np.arange(0,num_trials,51)
    condition = df.loc[df['sub']==s]['Trial_type'].values
    accu = df.loc[df['sub']==s]['trial_Corr'].values #accuracy
    EDS = 1*(condition=="EDS")
    IDS = 1*(condition=="IDS")
    Stay = 1*(condition=="Stay")

    # -- reduce LSS data to just amplitude information and then convert back to nii format
    # LSS data dim 4 set up like ...
    # [amplitude_trial_0, derivative_trial_0, amplitude_trial_1, derivative_trial_1, ...]
    cur_lss_nii = nib.load(nii_dir + "sub-%s/cue.LSS.nii.gz" %s)
    cur_lss_data = cur_lss_nii.get_fdata()
    lss_data_idx = np.array( [np.arange(0,num_trials*2,2), np.arange(1,num_trials*2,2)] ) #.flatten() # so that all amplitude values are in the 1st half and all derivative values are in the 2nd half of the file
    cur_lss_data_amp = cur_lss_data[:, :, :, lss_data_idx[0]]
    cur_lss_amp_nii=nilearn.image.new_img_like(cur_lss_nii, cur_lss_data_amp) # convert back to nii
    print("lss shape ",cur_lss_amp_nii.shape)

    brain_masker = nilearn.maskers.NiftiMasker()
    brain_time_series = brain_masker.fit_transform(cur_lss_amp_nii)

    for roi in [349]: #peak task rep coding ROI. remember to take minus one for indexing 
        
        #dervitaive variables
        ds = np.zeros(num_trials)

        # cal trial by trial derivatives
        for t1 in np.arange(num_trials-1):
            t2 = t1+1
            ds[t2] = data[roi-1, t1, t2]
        ds = np.sqrt(2*(1-abs(ds)))   #correlation distance
        
        #write this vector out and use it as amplitude regression in 3dDeconvolve
        ds[run_breaks] = 0
        np.savetxt(nii_dir + "sub-%s/switch_distance.txt" %s, ds)
        np.savetxt(nii_dir + "sub-%s/EDS_switch_distance.txt" %s, ds*EDS)
        np.savetxt(nii_dir + "sub-%s/IDS_switch_distance.txt" %s, ds*IDS)
        np.savetxt(nii_dir + "sub-%s/Stay_switch_distance.txt" %s, ds*Stay)


        #need to build X, include intercept
        X = np.array([np.ones(len(ds)),ds]).T

        #interaction with switching, no intercept
        X = np.array([EDS*ds, IDS*ds, Stay*ds]).T

        #remove run breaks
        X = np.delete(X, run_breaks, axis=0)
        brain_time_series = np.delete(brain_time_series, run_breaks, axis=0)

        #remove nans
        nans = np.unique(np.where(np.isnan(X))[0])
        if any(nans):
            X = np.delete(X, nans, axis=0)
            brain_time_series = np.delete(brain_time_series, nans, axis=0)

        #regress X = trial by trial distance, Y is trial by trial evoke
        results = linalg.lstsq(X, brain_time_series) #this is solving b in bX = brain_timeseries)

        for i, c in enumerate(["EDS", "IDS", "Stay"]):
            beta_nii = brain_masker.inverse_transform(results[0][i,:])
            beta_nii.to_filename(nii_dir + "sub-%s/%s_disntance_regression.nii.gz" %(s,c) )
