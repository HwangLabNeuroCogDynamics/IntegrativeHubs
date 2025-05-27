# code to do regression on RT and voxel distance
import numpy as np
import pandas as pd
from scipy.stats import zscore
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn 
import scipy.linalg as linalg
from joblib import Parallel, delayed
from nilearn.plotting import plot_stat_map
import os

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


# Function to process a single ROI, do disstance regression with a mixed effect model.
def ds_regression_roi_mixed(roi, data_dir, behav_df, subjects, model_syntax):
    # 1) collect each subject's small DataFrame in this list
    dfs = []

    for s in subjects:
        # --- load the big neural array, but only keep ds per trial-pair ---
        data = np.load(data_dir + "%s_ds.npz" %s)
        num_trials = data['ds'].shape[1]
        #run_breaks = np.arange(0, num_trials, 51)

        # behavioral vars for this subject
        sdf = behav_df.loc[behav_df['sub'] == s].reset_index(drop=True)
        subj_df = sdf.sort_values(by=['block', 'trial_n']).reset_index(drop=True)

        # compute ds (correlation distance) from previous trial ### ALREADY DONE in data
        # ds = np.zeros(num_trials)
        # for t in range(num_trials - 1):
        #     ds[t+1] = np.sqrt(2 * (1 - data[roi, t, t+1]))
        ds = data['ds'][roi, :]  # this is already the distance from the previous trial

        # figure out which trials are in the behavioral DataFrame
        trial_vec = []
        for i in range(len(subj_df)):
            trial_vec.append((subj_df['block'][i]-1) * 51 + subj_df['trial_n'][i]) #trials that are actually in DF, some are already dropped?         
        subj_df['ds'] = ds[trial_vec]

        subj_df['response_repeat'] = 1.0*(subj_df['Subject_Respo'] == subj_df['prev_resp'])
        subj_df['task_repeat'] = 1.0*(subj_df['Task'] == subj_df['prev_task'])
        # drop run breaks and any NaNs
        # Ensure categorical columns are treated as such:
        for col in ["Trial_type", "response_repeat", "task_repeat", "prev_target_feature_match"]:
            subj_df[col] = subj_df[col].astype("category")

        #remove firt trial, and only keep correct trials, overly slow trials, and post error trials
        subj_df = subj_df[subj_df['trial_n'] != 0]
        subj_df = subj_df[subj_df['trial_Corr'] != 0]
        subj_df = subj_df[subj_df['zRT'] <= 3]
        subj_df = subj_df[subj_df['prev_accuracy'] != 0]

        #append this subject's DataFrame to the list
        dfs.append(subj_df)

    # 2) concatenate all subjects
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna()  # Remove rows with missing values
    all_df = all_df.reset_index(drop=True)
    # 3) fit a single mixed‐effects model (random intercept by subject)
    md = smf.mixedlm( model_syntax, all_df, groups=all_df["sub"])
    mdf = md.fit(method="bfgs")

    # 4) extract the fixed‐effect estimates into a list of dicts
    results = []
    for name, coef, tval, pval in zip(
        mdf.params.index,
        mdf.params.values,
        mdf.tvalues,
        mdf.pvalues
    ):
        results.append({
            'ROI': roi,
            'Regressor': name,
            'Coefficient': coef,
            'T-value': tval,
            'P-value': pval
        })

    # 5) we also need to compute customized contrasts bewteen EDS and IDS,
    #get the fixed‐effect names and parameter vector
    fe_names = mdf.fe_params.index.tolist()
    fe_vals  = mdf.fe_params.values
    k_fe     = len(fe_names)

    # look up positions of your two Trial_type coefficients
    try:
        pos_eds = fe_names.index("C(Trial_type, Treatment(reference='Stay'))[T.EDS]")
        pos_ids = fe_names.index("C(Trial_type, Treatment(reference='Stay'))[T.IDS]")
    except ValueError:
        # one or both terms not in the model: skip the contrast
        pass
    else:
        #  build the contrast L: +1 for EDS, -1 for IDS
        L = np.zeros((1, k_fe))
        L[0, pos_eds] =  1
        L[0, pos_ids] = -1

        # run the Wald‐style t_test on the fixed effects
        ct = mdf.t_test(L)

        # extract numerical values
        # effect = β_EDS - β_IDS
        diff_coef = fe_vals[pos_eds] - fe_vals[pos_ids]
        tval      = np.squeeze(ct.tvalue)
        pval      = np.squeeze(ct.pvalue)

        results.append({
            'ROI':        roi,
            'Regressor':  'EDS_vs_IDS',
            'Coefficient': diff_coef,
            'T-value':     tval,
            'P-value':     pval
        })

    return results


########################
# RT switch cost analaysis
########################

#load behavioral data
data_dir = '/home/kahwang/argon/data/ThalHi/GLMsingle/trialwiseRSA/rdms_correlation_whitened_betas/'
nii_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/"
output_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/trialwiseRSA/"

#this is I gather that is from Xitong's fixes
#df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv") # this includes unusable subjects, totoal 74
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub'] #this is usable subjects after Steph's filtering.

#this is from Steph's coding, 59 subjects
behav_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ThalHi_MRI_zRTs_full.csv")

## looks like need to code task, response, feature repeat.
behav_df['response_repeat'] = 1.0*(behav_df['Subject_Respo'] == behav_df['prev_resp'])
behav_df['task_repeat'] = 1.0*(behav_df['Task'] == behav_df['prev_task'])

# Ensure categorical columns are treated as such:
for col in ["Trial_type", "response_repeat", "task_repeat", "prev_target_feature_match"]:
    behav_df[col] = behav_df[col].astype("category")

#remove firt trial, and only keep correct trials, overly slow trials, and post error trials
behav_df = behav_df[behav_df['trial_n'] != 0]
behav_df = behav_df[behav_df['trial_Corr'] != 0]
behav_df = behav_df[behav_df['zRT'] <= 3]
behav_df = behav_df[behav_df['prev_accuracy'] != 0]
behav_df['perceptual_change_c'] = behav_df['perceptual_change'] - behav_df['perceptual_change'].mean()

md = smf.mixedlm( "zRT ~C(Trial_type, Treatment(reference='IDS')) * perceptual_change_c +  " \
"C(response_repeat) * C(task_repeat) + C(task_repeat)* C(prev_target_feature_match , Treatment(reference='switch_target_feature'))" \
 "+ C(response_repeat) * C(prev_target_feature_match , Treatment(reference='switch_target_feature'))"  ,
                  behav_df, groups=behav_df["sub"], re_formula="~ 1 + C(Trial_type, Treatment(reference='IDS')) + C(task_repeat) + perceptual_change_c " )
mdf = md.fit(method="bfgs")
print(mdf.summary())


########################
# Run distance regression
########################

# first calculate trial by trial voxel distance, and save out the data, more efficient this way.
def batch_compute_ds_npz(subjects, data_dir, n_roi=418):
    for s in subjects:
        arr = np.load(os.path.join(data_dir, f"{s}_Schaefer400_Morel_BG_rdm_corr_whitened_resRT.npy"))
        # arr.shape == (n_roi, n_trials, n_trials)
        n_trials = arr.shape[2]

        # diagonal with offset=1 gives shape (n_roi, n_trials-1)
        off1 = np.diagonal(arr, axis1=1, axis2=2, offset=1)

        # build full ds array, pad t=0 with zeros
        ds = np.zeros((n_roi, n_trials), dtype=off1.dtype)
        #tempo = 1-off1 
        ds[:, 1:] = np.sqrt(2 * (1 - off1))  # if the coeff is Pearson correlation, this converts to distance

        fname = os.path.join(data_dir, f"{s}_ds.npz")
        np.savez_compressed(fname, ds=ds)
        print(f"✔ Subject {s}: saved ds shape {ds.shape} → {fname}")

batch_compute_ds_npz(subjects, data_dir)

df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ThalHi_MRI_zRTs_full.csv")

# now this is the model parallels the RT analysis
model_syntax = "ds ~ zRT*C(Trial_type, Treatment(reference='Stay')) + perceptual_change +  C(response_repeat) + C(task_repeat) + C(prev_target_feature_match , Treatment(reference='switch_target_feature'))"
results_list = Parallel(n_jobs=32)(delayed(ds_regression_roi_mixed)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)

for reg in results_df['Regressor'].unique():
    sub_df = results_df[results_df['Regressor'] == reg].sort_values('ROI')
    coef = sub_df['Coefficient'].values
    tval = sub_df['T-value'].values

    reg_name = reg.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_").replace("/", "_")
    write_stats_to_vol_yeo_template_nifti(coef, output_dir + "%s_coefficient.nii.gz" %reg_name)
    write_stats_to_vol_yeo_template_nifti(tval,  output_dir + "%s_tstat.nii.gz" %reg_name)


