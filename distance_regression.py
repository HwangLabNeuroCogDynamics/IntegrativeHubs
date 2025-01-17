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
from joblib import Parallel, delayed

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


# Function to process a single ROI, do distance regression
def ds_regression_roi(roi, data_dir, df, subjects):
    results = []
    subs_df = []
    
    for s in subjects:
        # Load neural correlation distance
        data = np.load(data_dir + f"{s}_whole_brain_coef.npy")  # ROI by trial by trial

        num_trials = data.shape[2]
        run_breaks = np.arange(0, num_trials, 51)
        condition = df.loc[df['sub'] == s]['Trial_type'].values
        accu = df.loc[df['sub'] == s]['trial_Corr'].values  # Accuracy
        accu[np.isnan(accu)] = 0
        RT = df.loc[df['sub'] == s]['rt'].values
        EDS = 1 * (condition == "EDS")
        IDS = 1 * (condition == "IDS")
        Stay = 1 * (condition == "Stay")

        # Response and cue repeat
        sdf = df.loc[df['sub'] == s].reset_index()
        cue = np.zeros(num_trials)
        resp = np.zeros(num_trials)
        for t1 in np.arange(num_trials - 1):
            t2 = t1 + 1
            if sdf.loc[t2, :]['cue'] == sdf.loc[t1, :]['cue']:
                cue[t2] = 1
            if sdf.loc[t2, :]['Subject_Respo'] == sdf.loc[t1, :]['Subject_Respo']:
                resp[t2] = 1                

        # Derivative of neural distance
        ds = np.zeros(num_trials)
        for t1 in np.arange(num_trials - 1):
            t2 = t1 + 1
            ds[t2] = data[roi - 1, t1, t2]
        ds = np.sqrt(2 * (1 - ds))  # Correlation distance

        # Build DataFrame for regression
        ds_df = pd.DataFrame({
            'ds': ds,
            'EDS': EDS,
            'IDS': IDS,
            'Stay': Stay,
            'RT': RT,
            'Accu': accu,
            'Cue': cue,
            'Response': resp,
            'subject': s,
            'ROI': roi
        })
        ds_df = ds_df.drop(index=run_breaks).reset_index(drop=True)  # Drop run breaks
        subs_df.append(ds_df)
    
    # Combine data for all subjects
    subs_df = pd.concat(subs_df, ignore_index=True).dropna()

    # Define and fit the mixed-effects model
    model_syntax = "RT ~ 1 + EDS*ds + IDS*ds + Cue + Response"
    random_effects_syntax = "1"
    model = smf.mixedlm(
        formula=model_syntax,
        data=subs_df,
        groups=subs_df["subject"],
        re_formula=random_effects_syntax
    ).fit()

    # Extract regression results
    for param, coef, tval, pval in zip(model.params.index, model.params, model.tvalues, model.pvalues):
        results.append({
            'ROI': roi,
            'Regressor': param,
            'Coefficient': coef,
            'T-value': tval,
            'P-value': pval
        })

    return results

# Main loop to parallelize across ROIs
results_list = Parallel(n_jobs=24)(  # Use all available CPU cores
    delayed(ds_regression_roi)(roi, data_dir, df, subjects) for roi in np.arange(418)
)

# Flatten the list of lists
results_list = [item for sublist in results_list for item in sublist]

# Create a DataFrame from the results
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression.csv", index=False)


# Function to extract and write t-values for the EDS:ds regressor
def create_t_value_nifti(results_df, output_filename, param, roisize=418):
    """
    Create a NIfTI file with the t-values for the EDS:ds regressor.
    
    Parameters:
    - results_df: DataFrame containing regression results.
    - output_filename: Path to save the NIfTI file.
    - roisize: Number of ROIs in the template.
    """
    # Extract t-values for the EDS:ds regressor
    eds_ds_t_values = np.zeros(roisize)  # Initialize with zeros
    for roi in range(roisize):
        t_value = results_df.loc[
            (results_df['ROI'] == roi) & (results_df['Regressor'] == param), 'T-value'
        ]
        if not t_value.empty:
            eds_ds_t_values[roi] = t_value.values[0]  # Assign the t-value for the ROI

    # Write the t-values to a NIfTI file using the provided function
    write_stats_to_vol_yeo_template_nifti(eds_ds_t_values, output_filename, roisize=roisize)

#Make how brain plots
for param in ['EDS', 'ds', 'EDS:ds', 'IDS', 'IDS:ds', 'Cue', 'Response']:
     create_t_value_nifti(results_df=results_df, output_filename="/home/kahwang/bin/IntegrativeHubs/data/%s_tvalues.nii.gz" %param, param = param)

#make parameter plots





####################
# graveyard
###############
    # #work on mix effect across subjects
    # # use RT and accu as DVs?

    # #write this vector out and use it as amplitude regression in 3dDeconvolve
    # ds[run_breaks] = 0
    # np.savetxt(nii_dir + "sub-%s/switch_distance.txt" %s, ds)
    # np.savetxt(nii_dir + "sub-%s/EDS_switch_distance.txt" %s, ds*EDS)
    # np.savetxt(nii_dir + "sub-%s/IDS_switch_distance.txt" %s, ds*IDS)
    # np.savetxt(nii_dir + "sub-%s/Stay_switch_distance.txt" %s, ds*Stay)


    # #need to build X, include intercept
    # X = np.array([np.ones(len(ds)),ds]).T

    # #interaction with switching, no intercept
    # X = np.array([EDS*ds, IDS*ds, Stay*ds]).T

    # #remove run breaks
    # X = np.delete(X, run_breaks, axis=0)
    # brain_time_series = np.delete(brain_time_series, run_breaks, axis=0)

    # #remove nans
    # nans = np.unique(np.where(np.isnan(X))[0])
    # if any(nans):
    #     X = np.delete(X, nans, axis=0)
    #     brain_time_series = np.delete(brain_time_series, nans, axis=0)

    # #regress X = trial by trial distance, Y is trial by trial evoke
    # results = linalg.lstsq(X, brain_time_series) #this is solving b in bX = brain_timeseries)

    # for i, c in enumerate(["EDS", "IDS", "Stay"]):
    #     beta_nii = brain_masker.inverse_transform(results[0][i,:])
    #     beta_nii.to_filename(nii_dir + "sub-%s/%s_disntance_regression.nii.gz" %(s,c) )
