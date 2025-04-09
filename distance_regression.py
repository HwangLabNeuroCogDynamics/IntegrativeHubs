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
            ds[t2] = data[roi, t1, t2] #the roi is zero based here
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
            for param in model.params.index:  # Update this list as needed
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
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat "
results_list = Parallel(n_jobs=24)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
# Flatten the list of lists
results_list = [item for sublist in results_list for item in sublist]
# Create a DataFrame from the results
results_df = pd.DataFrame(results_list)
# Save the results to a CSV file
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_main_effect.csv", index=False)
#results_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_main_effect.csv")


########################
# Model 1.1, cue and task switch effect?
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS:Task_repeat + IDS:Task_switch + EDS:Task_repeat + EDS:Task_switch + Response_repeat"
results_list = Parallel(n_jobs=16)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_cue_task_switch.csv", index=False)


########################
# Model 1.2, cue and task switch effect? as well as response repeat?
########################
model_syntax = (
    "ds ~ 0 + Stay:Cue_repeat:Response_repeat + "
    "Stay:Cue_switch:Response_repeat + "
    "IDS:Task_switch:Response_repeat + "
    "EDS:Task_repeat:Response_repeat + "
    "EDS:Task_switch:Response_repeat + "
    "Stay:Cue_repeat:Response_switch + "
    "Stay:Cue_switch:Response_switch + "
    "IDS:Task_switch:Response_switch + "
    "EDS:Task_repeat:Response_switch + "
    "EDS:Task_switch:Response_switch "
)
results_list = Parallel(n_jobs=20)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_cue_task_response_switch.csv", index=False)


########################
# Model 2, understand the effect of accuracy
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Stay:Accu + IDS:Accu + EDS:Accu + Stay:Cue_repeat:Accu + Stay:Cue_switch:Accu"
results_list = Parallel(n_jobs=10)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)


########################
# Model 3, understand the effect of RT
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat + Stay:Cue_repeat:RT + Stay:Cue_switch:RT + IDS:RT + EDS:RT"
results_list = Parallel(n_jobs=8)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
results_list = [item for sublist in results_list for item in sublist]
results_df = pd.DataFrame(results_list)
results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_rt_effect.csv", index=False)


########################
# Just calculate ds and save it, maybe for parametric regression latter
########################


def save_ds_files_by_roi(data_dir, nii_dir, df, subjects, roi_list):
    """
    For each subject, compute the ds values (correlation distance) for each ROI,
    adjust run-break trials to have ds = 0, compute condition multipliers from the df,
    and then save the ds values into text files.
    
    The function creates the following files (inside a subject-specific directory):
        - switch_distance.txt: The ds array for each trial (rows) and ROI (columns).
        - EDS_switch_distance.txt: The ds array multiplied elementwise by EDS (binary vector).
        - IDS_switch_distance.txt: The ds array multiplied elementwise by IDS.
        - Stay_switch_distance.txt: The ds array multiplied elementwise by Stay.
    
    Parameters:
        data_dir (str): Directory containing the subject npy files.
        nii_dir (str): Base directory to save the text files.
        df (pd.DataFrame): DataFrame containing subject information and condition labels.
                           It must have a column 'sub' (subject ID) and a column 'Trial_type'.
        subjects (list or pd.Series): List or Series of subject identifiers.
        roi_list (list): List of ROI indices (1-indexed) for which ds values are computed.
    
    Returns:
        None
    """
    # If subjects is a pandas Series, convert it to a list.
    if isinstance(subjects, pd.Series):
        subjects = subjects.tolist()

    # Process each subject
    for s in subjects:
        # Build the file path and load the subject's whole-brain data.
        # Data is expected to have shape: (n_rois, n_trials, n_trials)
        file_path = os.path.join(data_dir, f"{s}_whole_brain_coef.npy")
        data = np.load(file_path)
        num_trials = data.shape[2]

        # Define run-break trial indices (e.g., every 51st trial)
        run_breaks = np.arange(0, num_trials, 51)

        # Compute ds for each ROI and store in a list.
        ds_roi_list = []
        for roi in roi_list:
            ds = np.zeros(num_trials)
            # Compute ds using the "previous" trial (t-1 to t)
            for t in range(1, num_trials):
                # Note: roi is assumed to be 1-indexed; adjust to 0-indexed
                ds[t] = data[roi - 1, t - 1, t]
            # Convert correlation values to correlation distance
            ds = np.sqrt(2 * (1 - ds))
            # Instead of dropping run-break trials, set their ds values to 0.
            ds[run_breaks] = 0
            ds_roi_list.append(ds)

        # Stack the ds vectors column-wise so that each row is a trial and each column is a ROI.
        # Resulting shape: (n_trials, n_rois)
        ds_array = np.column_stack(ds_roi_list)

        # --- Retrieve condition information from df ---
        # Expect df to have a 'Trial_type' column and a 'sub' column that matches s.
        subject_df = df.loc[df['sub'] == s].reset_index(drop=True)
        # Ensure that the number of trials in the DataFrame matches ds_array
        if len(subject_df) != num_trials:
            raise ValueError(f"Subject {s}: number of rows in df ({len(subject_df)}) "
                             f"does not match number of trials ({num_trials}).")
        
        condition = subject_df['Trial_type'].values  # shape: (n_trials,)
        # Create binary arrays for each condition.
        EDS = (condition == "EDS").astype(int)   # 1 when condition is "EDS", else 0.
        IDS = (condition == "IDS").astype(int)   # 1 when condition is "IDS", else 0.
        Stay = (condition == "Stay").astype(int)   # 1 when condition is "Stay", else 0.
        
        # Multiply ds_array (2D: trials x ROIs) by the condition vectors (reshaped to be 2D)
        ds_EDS  = ds_array * EDS[:, np.newaxis]
        ds_IDS  = ds_array * IDS[:, np.newaxis]
        ds_Stay = ds_array * Stay[:, np.newaxis]

        # --- Save the arrays to text files ---
        # Create a subject-specific directory if it doesn't exist.
        sub_dir = os.path.join(nii_dir, f"sub-{s}")
        os.makedirs(sub_dir, exist_ok=True)

        # Save each array using np.savetxt.
        np.savetxt(os.path.join(sub_dir, "switch_distance.txt"), ds_array, fmt="%.6f")
        np.savetxt(os.path.join(sub_dir, "EDS_switch_distance.txt"), ds_EDS, fmt="%.6f")
        np.savetxt(os.path.join(sub_dir, "IDS_switch_distance.txt"), ds_IDS, fmt="%.6f")
        np.savetxt(os.path.join(sub_dir, "Stay_switch_distance.txt"), ds_Stay, fmt="%.6f")

        print(f"Saved ds files for subject {s} in {sub_dir}")

# Example usage:
# Assuming you have the variables: data_dir, nii_dir, df, subjects, and roi_list defined,
# you can call:
#
# save_ds_files_by_roi(data_dir, nii_dir, df, subjects, roi_list)


ds_by_subject = save_ds_files_by_roi(data_dir, nii_dir, df, subjects, roi_list)

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
