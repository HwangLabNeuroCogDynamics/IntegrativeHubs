# code to test if trail evoke response correlate with neural distance (effort for switching)
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

# Function to process a single ROI, do distance regression
def ds_regression_roi_per_subject(roi, data_dir, df, subjects, model_syntax):
    results = []

    for s in subjects:
        # Load neural correlation distance
        data = np.load(data_dir + f"{s}_whole_brain_coef.npy")  # ROI by trial by trial

        num_trials = data.shape[2]
        run_breaks = np.arange(0, num_trials, 51)
        condition = df.loc[df['sub'] == s]['Trial_type'].values
        accu = df.loc[df['sub'] == s]['trial_Corr'].values  # Accuracy
        accu[np.isnan(accu)] = 0
        RT = df.loc[df['sub'] == s]['rt'].values
        RT = RT - np.nanmean(RT) #center
        EDS = 1 * (condition == "EDS")
        IDS = 1 * (condition == "IDS")
        Stay = 1 * (condition == "Stay")

        # Response, task, and cue repeat
        sdf = df.loc[df['sub'] == s].reset_index()
        cue = np.zeros(num_trials)
        resp = np.zeros(num_trials)
        task = np.zeros(num_trials)
        for t1 in np.arange(num_trials - 1):
            t2 = t1 + 1
            if sdf.loc[t2, :]['cue'] == sdf.loc[t1, :]['cue']:
                cue[t2] = 1
            if sdf.loc[t2, :]['Subject_Respo'] == sdf.loc[t1, :]['Subject_Respo']:
                resp[t2] = 1                
            if sdf.loc[t2, :]['Task'] == sdf.loc[t1, :]['Task']:
                task[t2] = 1     

        # Derivative of neural distance
        ds = np.zeros(num_trials)
        for t1 in np.arange(num_trials - 1):  # make sure it is distance from the "previous" trial
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
            'Error': 1*(accu!=1),
            'Response_repeat': resp,
            'Response_switch': 1*(resp!=1),
            'Task_repeat': task,
            'Task_switch': 1*(task!=1),
            'Cue_repeat': cue,
            'Cue_switch': 1*(cue!=1),
        })
        ds_df = ds_df.drop(index=run_breaks).reset_index(drop=True)  # Drop run breaks
        ds_df = ds_df.dropna()  # Remove rows with missing values

        #run model
        try:
            model = smf.ols(formula=model_syntax, data=ds_df).fit()

            # Extract regression results
            for param, coef, tval, pval in zip(model.params.index, model.params, model.tvalues, model.pvalues):
                results.append({
                    'ROI': roi,
                    'Subject': s,
                    'Regressor': param,
                    'Coefficient': coef,
                    'T-value': tval,
                    'P-value': pval
                })
        except Exception as e:
            # Insert NaNs for this subject and ROI in case of an error
            for param in ["Stay:Accu:RT", "EDS:Accu:RT", "IDS:Accu:RT", "Cue", "Response"]:  # Update this list as needed
                results.append({
                    'ROI': roi,
                    'Subject': s,
                    'Regressor': param,
                    'Coefficient': float('nan'),
                    'T-value': float('nan'),
                    'P-value': float('nan')
                })

    return results

# Function to perform second-level statistics

def group_level_stats_flexible(results_df, param1, param2=None, roisize=418):
    """
    Perform group-level statistics (one-sample t-test or paired t-test) for given regressors.

    Parameters:
    - results_df: DataFrame containing regression results.
    - param1: Name of the first regressor to analyze (always used).
    - param2: Name of the second regressor for comparison (optional).
    - roisize: Number of ROIs in the template.

    Returns:
    - group_coefficients: Array of group-level mean coefficients for param1 (if param2 is None).
    - group_t_values: Array of group-level t-values (one-sample or paired, based on param2).
    """
    group_coefficients = np.zeros(roisize)
    group_t_values = np.zeros(roisize)

    for roi in range(roisize):
        if param2 is None:
            # One-sample t-test for param1 against 0
            roi_coefficients = results_df.loc[
                (results_df['ROI'] == roi) & (results_df['Regressor'] == param1), 'Coefficient'
            ].dropna()

            if not roi_coefficients.empty:
                t_stat, _ = ttest_1samp(roi_coefficients, popmean=0, nan_policy='omit')
                group_coefficients[roi] = roi_coefficients.mean()
                group_t_values[roi] = t_stat

        else:
            # Paired t-test between param1 and param2
            roi_coeff1 = results_df.loc[
                (results_df['ROI'] == roi) & (results_df['Regressor'] == param1), 'Coefficient'
            ].dropna()
            roi_coeff2 = results_df.loc[
                (results_df['ROI'] == roi) & (results_df['Regressor'] == param2), 'Coefficient'
            ].dropna()

            if not roi_coeff1.empty and not roi_coeff2.empty and len(roi_coeff1) == len(roi_coeff2):
                t_stat, _ = ttest_rel(roi_coeff1, roi_coeff2)
                group_coefficients[roi] = roi_coeff1.mean() - roi_coeff2.mean()  # Mean difference
                group_t_values[roi] = t_stat

    return group_coefficients, group_t_values

def write_group_stats_to_nifti(coefficients, t_values, output_prefix, roisize=418):
    """
    Write group-level coefficients and t-values to NIfTI files.

    Parameters:
    - coefficients: Array of group-level coefficients.
    - t_values: Array of group-level t-values.
    - output_prefix: Prefix for the output NIfTI files.
    - roisize: Number of ROIs in the template.
    """
    # Write coefficients
    coef_nifti_filename = f"{output_prefix}_coefficients.nii.gz"
    write_stats_to_vol_yeo_template_nifti(coefficients, coef_nifti_filename, roisize)

    # Write t-values
    tval_nifti_filename = f"{output_prefix}_tvalues.nii.gz"
    write_stats_to_vol_yeo_template_nifti(t_values, tval_nifti_filename, roisize)

    return coef_nifti_filename, tval_nifti_filename

def plot_nifti_with_nilearn(nifti_file, title):
    """
    Plot a NIfTI file using Nilearn.

    Parameters:
    - nifti_file: Path to the NIfTI file to plot.
    - title: Title for the plot.
    """
    plot_stat_map(nifti_file, display_mode="z", cut_coords=5, title=title)


#load behavioral data
data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/coefs/'
nii_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/"

df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']

########################
# Run main effect model
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS"
results_list = Parallel(n_jobs=16)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
# Flatten the list of lists
results_list = [item for sublist in results_list for item in sublist]
# Create a DataFrame from the results
results_df = pd.DataFrame(results_list)
# Save the results to a CSV file
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_main_effect.csv", index=False)

# group stats, key contrasts are EDS v IDS and IDS v Stay, eventually, look into cue repeat and resposnse repeat?
param = 'EDS'
group_coefficients, group_t_values = group_level_stats_flexible(results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param1 = 'EDS'
param2 = 'IDS'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'IDS'
param2 = 'Stay:Cue_switch'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'Stay:Cue_switch'
param2 = 'Stay:Cue_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")


########################
# Model 1.1, cue and task switch effect?
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS:Task_repeat + IDS:Task_switch + EDS:Task_repeat + EDS:Task_switch + Response_repeat"
results_list = Parallel(n_jobs=16)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_cue_task_switch.csv", index=False)

param1 = 'EDS:Task_switch'
param2 = 'EDS:Task_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_repeat'
param2 = 'IDS:Task_switch'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_repeat'
param2 = 'Stay:Cue_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

########################
# Model 2, understand the effect of accuracy
########################

model_syntax = "ds ~ 0 + Stay + IDS + EDS + Stay:Accu + IDS:Accu + EDS:Accu"
results_list = Parallel(n_jobs=16)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_accu_effect.csv", index=False)

#####
param1 = 'EDS:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")

param1 = 'IDS:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")

param1 = 'Stay:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")



########################
# Model 3, understand the effect of RT
########################

model_syntax = "ds ~ 0 + Stay + IDS + EDS + Stay:RT + IDS:RT + EDS:RT"
results_list = Parallel(n_jobs=16)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_rt_effect.csv", index=False)

param = 'EDS:RT'
group_coefficients, group_t_values = group_level_stats_flexible(results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'IDS:RT'
group_coefficients, group_t_values = group_level_stats_flexible(results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Stay:RT'
group_coefficients, group_t_values = group_level_stats_flexible(results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")




#### how to project data to surface? Looks nicer that way




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
