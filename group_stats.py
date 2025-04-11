# run group stats on distance measures
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
from itertools import product
from scipy import stats
from scipy.ndimage import label


# first load all the EEG group stats and organize them
ROOT_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/ThalHi_data/"
EEG_subs = np.array([128, 112, 108, 110, 120,  98,  86,  82, 115,  94,  76,  91,  95,
       121, 114, 125, 107, 111,  88, 113, 131, 130, 135, 140, 167, 145,
       146, 138, 147, 176, 122, 118, 103, 142])

for model in ["main_effect_model", "switch_model", "response_switch_model", "RT_model", "accuracy_model"]:
    sub_dfs = []
    for sub in EEG_subs:
        file_path = ROOT_DIR + "RSA/distance_results/" + str(sub) + "_%s_regression_results.csv" % model
        sub_df = pd.read_csv(file_path)
        sub_dfs.append(sub_df)
        subs_df = pd.concat(sub_dfs, ignore_index=True)
    subs_df.to_csv(ROOT_DIR + "RSA/distance_results/all_subs_%s_regression_results.csv" % model, index=False)


def matrix_permutation(M1, M2, threshold, p, n_perm=5000, ttest=True):
    """
    Randomized permutation test for element-wise comparison.
    
    This function assumes that M1 and M2 are 3D arrays with shape
      (num_subjects, T1, T2).
    
    Parameters
    ----------
    M1 : np.ndarray
        Coefficient matrix for condition 1.
    M2 : np.ndarray
        Coefficient matrix for condition 2.
        (If doing a one-sample test, set M2 to an array of zeros of the same shape as M1.)
    threshold : float
        t–value threshold for cluster formation.
    p : float
        Significance level (e.g. 0.05).
    n_perm : int, optional
        Number of permutation iterations (default 5000).
    ttest : bool, optional
        If True, perform a one-sample t-test on the differences; otherwise compute the mean difference.
    
    Returns
    -------
    mask : np.ndarray
        A binary mask (2D array, shape = [T1, T2]) that is 1 where significant clusters are found.
    mat : np.ndarray
        The t–value matrix computed from the actual data.
    """
    if M1.shape[0] != M2.shape[0]:
        print("Check matrix dimensions")
        return
    num_sub = M1.shape[0]
    # Compute subject-wise differences.
    dM = M1 - M2

    # Collect the maximum cluster mass from each permutation
    null_mass = np.zeros(n_perm)
    for iter in np.arange(n_perm):
        # Create random sign flips for each subject (+1 or -1)
        rand_vec = (2 * np.random.randint(0, 2, size=num_sub) - 1)
        permuted_M = np.empty_like(dM)
        for i, r in enumerate(rand_vec):
            permuted_M[i, :, :] = dM[i, :, :] * r

        # Compute t–value (or mean) for this permutation
        if ttest:
            # Note: Using axis=0 so that we test across subjects.
            permuted_t = stats.ttest_1samp(permuted_M, 0, axis=0, nan_policy='omit')[0]
        else:
            permuted_t = np.nanmean(permuted_M, axis=0)

        mass = 0
        # For positive clusters:
        cmat, num_clust = label(permuted_t > threshold)
        if num_clust > 0:
            for c in range(1, num_clust+1):
                temp_mass = np.sum(permuted_t[cmat == c])
                if temp_mass > mass:
                    mass = temp_mass
        # For negative clusters:
        cmat, num_clust = label(permuted_t < -threshold)
        if num_clust > 0:
            for c in range(1, num_clust+1):
                temp_mass = np.sum(permuted_t[cmat == c])
                if abs(temp_mass) > mass:
                    mass = abs(temp_mass)
        null_mass[iter] = mass

    # Determine cluster mass thresholds from the permutation null distribution.
    pos_mass_thresh = np.quantile(null_mass, 1 - 0.5 * p)
    neg_mass_thresh = np.quantile(null_mass, 1 - 0.5 * p)

    # Compute actual t–values on the original (un-permuted) differences.
    if ttest:
        mat = stats.ttest_1samp(dM, 0, axis=0, nan_policy='omit')[0]
    else:
        mat = np.nanmean(dM, axis=0)

    # Form clusters and flag significant ones
    mask = np.zeros(mat.shape)
    # Positive clusters:
    cmat, num_clust = label(mat > threshold)
    if num_clust > 0:
        for c in range(1, num_clust+1):
            if np.sum(mat[cmat == c]) > pos_mass_thresh:
                mask += (cmat == c)
    # Negative clusters:
    cmat, num_clust = label(mat < -threshold)
    if num_clust > 0:
        for c in range(1, num_clust+1):
            if abs(np.sum(mat[cmat == c])) > neg_mass_thresh:
                mask += (cmat == c)
    return mask, mat

def compute_t_coef_grid_cluster(subs, subs_df, param1, param2=None,
                                step=10, max_val=1940, thresh=2.5, p=0.05, n_perm=5000):
    """
    Compute t–value and coefficient grids from regression results via a cluster-based permutation test.
    
    For each subject, the function loads a CSV file containing regression results and
    pivots the data into a matrix (T1 x T2) for the specified regressor(s). It then computes:
      - The average coefficient (or difference between coefficients)
      - A t–value grid (via one-sample or paired t-test across subjects)
      - A binary mask of significant clusters (obtained via permutation testing)
    
    Parameters
    ----------
    subs : list
        List of subject identifiers. Each subject is assumed to have a file
        named: "<path_prefix><sub>_main_effect_model_regression_results.csv".
    param1 : str
        Name of the primary regressor.
    param2 : str, optional
        Name of the second regressor. If provided, a paired t-test is performed
        (and the output coefficient grid is the difference in means); otherwise,
        a one-sample t-test against zero is performed.
    path_prefix : str, optional
        Directory path where the CSV files reside.
    step : int, optional
        Step size for sampling time points (both T1 and T2).
    max_val : int, optional
        Maximum value for the time range (non-inclusive).
    thresh : float, optional
        t–value threshold for forming clusters.
    p : float, optional
        Significance level for the permutation test.
    n_perm : int, optional
        Number of permutation iterations.
    
    Returns
    -------
    t_grid : np.ndarray
        The t–value grid (2D array) computed from the actual data.
    coef_grid : np.ndarray
        A grid (2D array) of average coefficients (or differences) across subjects.
    cluster_mask : np.ndarray
        A binary mask (2D array) indicating significant clusters.
    """
    # Load and concatenate all subject CSV files.
    # sub_dfs = []
    # for sub in subs:
    #     file_path = f'{path_prefix}{sub}_main_effect_model_regression_results.csv'
    #     sub_df = pd.read_csv(file_path)
    #     sub_dfs.append(sub_df)
    # subs_df = pd.concat(sub_dfs, ignore_index=True)
    
    # Define the time points.
    t_points = list(range(0, max_val, step))
    nT = len(t_points)
    num_sub = len(subs)
    
    # Create 3D arrays for each regressor.
    # M1 (and M2, if needed) will be organized as (subject, T1, T2)
    M1 = np.full((num_sub, nT, nT), np.nan)
    if param2 is not None:
        M2 = np.full((num_sub, nT, nT), np.nan)
    else:
        # For one-sample test, we compare against zero.
        M2 = np.zeros((num_sub, nT, nT))
    
    # For each subject, pivot the data to extract a matrix for the regressor(s).
    for i, sub in enumerate(subs):
        df_sub = subs_df[subs_df['Subject'] == sub]
        # For param1:
        df_param1 = df_sub[df_sub['Regressor'] == param1]
        pivot1 = df_param1.pivot(index='T1', columns='T2', values='Coefficient')
        # Ensure that the pivoted table covers all desired time points.
        pivot1 = pivot1.reindex(index=t_points, columns=t_points)
        M1[i, :, :] = pivot1.values
        
        if param2 is not None:
            df_param2 = df_sub[df_sub['Regressor'] == param2]
            pivot2 = df_param2.pivot(index='T1', columns='T2', values='Coefficient')
            pivot2 = pivot2.reindex(index=t_points, columns=t_points)
            M2[i, :, :] = pivot2.values

    # Compute the coefficient grid:
    if param2 is None:
        # One-sample: average the coefficients across subjects.
        coef_grid = np.nanmean(M1, axis=0)
    else:
        # Paired test: difference between average coefficients.
        coef_grid = np.nanmean(M1, axis=0) - np.nanmean(M2, axis=0)
    
    # Run the cluster-based permutation test on the 3D data.
    # This returns a mask of significant clusters and the t–value grid.
    cluster_mask, t_grid = matrix_permutation(M1, M2, threshold=thresh, p=p,
                                               n_perm=n_perm, ttest=True)
    
    return t_grid, coef_grid, cluster_mask


# fMRI group stats function. Function to perform second-level statistics on fMRI data
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

def write_stats_to_vol_yeo_template_nifti(graph_metric, fn, roisize = 418):
	'''short hand to write vol based nifti file of the stats
	, voxels in each parcel will be replaced with the stats'''


	vol_template = nib.load('/mnt/nfs/lss/lss_kahwang_hpc/ROIs/Schaefer400+Morel+BG_2.5.nii.gz')
	v_data = vol_template.get_fdata()
	graph_data = np.zeros((np.shape(v_data)))

	for i in np.arange(roisize):
		graph_data[v_data == i+1] = graph_metric[i]  # roi indices start from 1

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn) 
	return new_nii

def visualize_t_grid(t_grid, cluster_mask,
                     output_npy='t_grid.npy',
                     output_img='t_grid_sig_clusters.png',
                     title='Significant Clusters Heatmap'):
    """
    Visualize the t–value grid, showing only cells corresponding to significant clusters.

    Parameters
    ----------
    t_grid : np.ndarray
        A 2D array containing t–values (from your cluster-based permutation test).
    cluster_mask : np.ndarray
        A binary (or integer) mask of the same shape as t_grid. Nonzero entries indicate
        significant clusters.
    output_npy : str, optional
        File name (or path) where the t_grid will be saved in NumPy format.
    output_img : str, optional
        File name (or path) where the heatmap image will be saved.
    title : str, optional
        Title for the heatmap.

    Actions
    -------
    - Saves the t_grid array (via np.save).
    - Generates a seaborn heatmap that visualizes only the significant clusters.
    - Displays and saves the resulting heatmap.
    """
    # Save the t_grid array so you can load it in the future.
    np.save(output_npy, t_grid)

    # Create a mask for seaborn heatmap: hide cells that are not part of any significant cluster.
    # Here, we assume that cluster_mask is nonzero for significant cells.
    heatmap_mask = (cluster_mask == 0)

    # Plot the heatmap of the t_grid, using the heatmap_mask.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(t_grid, mask=heatmap_mask, cmap="coolwarm", annot=False,
                     cbar_kws={'label': 't-value'})
    plt.title(title)
    plt.xlabel('T2')
    plt.ylabel('T1')
    plt.tight_layout()

    # Save the heatmap image.
    plt.savefig(output_img, dpi=300)
    plt.show()


# Example usage:
# After obtaining t_grid and cluster_mask from compute_t_coef_grid_cluster, you can call:
#
# t_grid, coef_grid, cluster_mask = compute_t_coef_grid_cluster(subs, 'ParamA', param2='ParamB')
# visualize_t_grid(t_grid, cluster_mask,
#                  output_npy='my_t_grid.npy',
#                  output_img='my_t_grid_sig_clusters.png',
#                  title='Significant Clusters (Permutation Test)')


########################
# Run main effect model
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat "
#results_list = Parallel(n_jobs=10)(delayed(ds_regression_roi_per_subject)(roi, data_dir, df, subjects, model_syntax) for roi in np.arange(418))
# Flatten the list of lists
#results_list = [item for sublist in results_list for item in sublist]
# Create a DataFrame from the results
#results_df = pd.DataFrame(results_list)
# Save the results to a CSV file
#results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_main_effect.csv", index=False)
fmri_results_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_main_effect.csv")
EEG_results_df = pd.read_csv(ROOT_DIR + "RSA/distance_results/all_subs_main_effect_model_regression_results.csv")


# fMRI group stats, key contrasts are EDS v IDS and IDS v Stay, eventually, look into cue repeat and resposnse repeat?
param = 'EDS'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'IDS'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Stay:Cue_repeat'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Stay:Cue_switch'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param1 = 'EDS'
param2 = 'IDS'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'IDS'
param2 = 'Stay:Cue_switch'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'IDS'
param2 = 'Stay:Cue_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")


param1 = 'Stay:Cue_switch'
param2 = 'Stay:Cue_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param = 'Response_repeat'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Task_repeat'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")


## EEG group stats
param1 = 'EDS'
param2 = 'IDS'
t_grid, coef_grid, cluster_mask = compute_t_coef_grid_cluster(EEG_subs, EEG_results_df, 'EDS', param2='IDS', thresh=2, p=0.05, n_perm=5000)
visualize_t_grid(t_grid, cluster_mask,
                 output_npy='/home/kahwang/bin/IntegrativeHubs/data/EEG_EDSvIDS_t_grid.npy',
                  output_img='/home/kahwang/bin/IntegrativeHubs/data/EEG_EDSvIDS_t_grid.png',
                 title='EDS - Stay T Grid Heatmap')
# visualize_t_grid(t_grid, threshold=2.5, output_npy='/home/kahwang/bin/IntegrativeHubs/data/EEG_EDSvIDS_t_grid.npy', output_img='/home/kahwang/bin/IntegrativeHubs/data/EEG_EDSvIDS_t_grid.png', 
#                  title='EDS - IDS T Grid Heatmap')

########################
# Model 1.1, cue and task switch effect?
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS:Task_repeat + IDS:Task_switch + EDS:Task_repeat + EDS:Task_switch + Response_repeat"
fmri_results_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_cue_task_switch.csv")

param1 = 'EDS:Task_switch'
param2 = 'EDS:Task_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_repeat'
param2 = 'IDS:Task_switch'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_switch'
param2 = 'IDS:Task_switch'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_repeat'
param2 = 'Stay:Cue_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

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
fmri_results_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_cue_task_response_switch.csv")

# rule switch effect
param1 = 'EDS:Task_repeat:Response_repeat'
param2 = 'Stay:Cue_repeat:Response_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_switch:Response_repeat'
param2 = 'IDS:Task_switch:Response_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

# clean response effect
param1 = 'Stay:Cue_repeat:Response_switch'
param2 = 'Stay:Cue_repeat:Response_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

#clean task switch effect
param1 = 'IDS:Task_switch:Response_repeat'
param2 = 'Stay:Cue_switch:Response_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")

param1 = 'EDS:Task_switch:Response_repeat'
param2 = 'EDS:Task_repeat:Response_repeat'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1, param2)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}_vs_{param2}")


########################
# Model 2, understand the effect of accuracy
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Stay:Accu + IDS:Accu + EDS:Accu + Stay:Cue_repeat:Accu + Stay:Cue_switch:Accu"
fmri_results_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_accu_effect.csv", index=False)

#####
param1 = 'EDS:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")

param1 = 'IDS:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")

param1 = 'Stay:Cue_switch:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")

param1 = 'Stay:Cue_repeat:Accu'
group_coefficients_diff, group_t_values_diff = group_level_stats_flexible(fmri_results_df, param1)
write_group_stats_to_nifti(group_coefficients_diff, group_t_values_diff, f"/home/kahwang/bin/IntegrativeHubs/data/{param1}")



########################
# Model 3, understand the effect of RT
########################

model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat + Stay:Cue_repeat:RT + Stay:Cue_switch:RT + IDS:RT + EDS:RT"
fmri_results_df = pd.read_csv("/home/kahwang/bin/IntegrativeHubs/data/ds_regression_rt_effect.csv")

param = 'EDS:RT'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'IDS:RT'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Stay:Cue_switch:RT'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")

param = 'Stay:Cue_repeat:RT'
group_coefficients, group_t_values = group_level_stats_flexible(fmri_results_df, param)
write_group_stats_to_nifti(group_coefficients, group_t_values, f"/home/kahwang/bin/IntegrativeHubs/data/{param}")