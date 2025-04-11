# Import necessary libraries
import os
import glob
import numpy as np
import pandas as pd
import scipy
from scipy.stats import zscore, ttest_1samp, ttest_rel
from scipy.special import logit
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import ( train_test_split, ShuffleSplit, cross_val_score, cross_val_predict, KFold, LeaveOneOut )
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_stat_map
import nibabel as nib
import mne
from mne.time_frequency import tfr_morlet
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime
import scipy.linalg as linalg
from itertools import product

#load behavioral data
ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
sub = input()  

this_sub_path=ROOT+ 'eeg_preproc_RespToReviewers/' +str(sub)
all_probe = mne.read_epochs(this_sub_path+'/probe events-epo.fif')
all_probe.baseline = None
# meta_df = pd.read_csv('/Shared/lss_kahwang_hpc/ThalHi_data/ThalHi_EEG_metadata.csv')
# tmp_df = meta_df.loc[meta_df['sub']==int(sub)]
# tmp_df = tmp_df.reset_index()
df = all_probe.metadata
df = df.reset_index()

coef_mat = np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_coef_mat.npy')
# in the shape of trial by time by time, but here trial t is actually trial + 1 given it is a distance measure from the previous trial
# make sure it is distance from the "previous" trial

num_tps = coef_mat.shape[1]  # Number of time points    
num_trials = df.shape[0]
run_breaks = np.arange(0, num_trials, 51)
condition = df['Trial_type'].values
accu = df['trial_Corr'].values  # Accuracy
accu[accu!=1] = 0
accu[np.isnan(accu)] = 0
RT = df['rt'].values
RT = np.array([float(val) if val != 'none' else np.nan for val in RT])
RT = RT - np.nanmean(RT) #center
EDS = 1 * (condition == "EDS")
IDS = 1 * (condition == "IDS")
Stay = 1 * (condition == "Stay")

# Response, task, and cue repeat
cue = np.zeros(num_trials)
resp = np.zeros(num_trials)
task = np.zeros(num_trials)
for t1 in np.arange(num_trials - 1):
    t2 = t1 + 1
    if df.loc[t2, :]['cue'] == df.loc[t1, :]['cue']:
        cue[t2] = 1
    if df.loc[t2, :]['Subject_Respo'] == df.loc[t1, :]['Subject_Respo']:
        resp[t2] = 1                
    if df.loc[t2, :]['Task'] == df.loc[t1, :]['Task']:
        task[t2] = 1    

### looks like need quite a lot of memory for this 
coef_mat[1:] = coef_mat[:-1]

# Now set the first slice to NaNs:
coef_mat[0].fill(np.nan)

##### some kind of statistics here...
#we will do it acroos the time x time grid. 
#ds = np.sqrt(2 * (1 - coef_mat))  # Correlation distance
ds = coef_mat
## within subject regression
def process_regression(t1, t2, model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + " "IDS:Task_repeat + IDS:Task_switch + " "EDS:Task_repeat + EDS:Task_switch + Response_repeat"):
    """
    Process a single (t1, t2) iteration: build the DataFrame,
    drop run breaks and NaNs, run the regression, and return a list of results.
    """
    neural_dist = ds[:, t1, t2]
    
    # Build DataFrame for regression
    ds_df = pd.DataFrame({
        'ds': neural_dist,
        'EDS': EDS,
        'IDS': IDS,
        'Stay': Stay,
        'RT': RT,
        'Accu': accu,
        'Error': 1 * (accu != 1),
        'Response_repeat': resp,
        'Response_switch': 1 * (resp != 1),
        'Task_repeat': task,
        'Task_switch': 1 * (task != 1),
        'Cue_repeat': cue,
        'Cue_switch': 1 * (cue != 1)
    })

    # Drop run breaks and rows with missing values
    ds_df = ds_df.drop(index=run_breaks).reset_index(drop=True)
    ds_df = ds_df.dropna()

    results = []
    try:
        model = smf.ols(formula=model_syntax, data=ds_df).fit()
        for param, coef, tval, pval in zip(model.params.index,
                                           model.params,
                                           model.tvalues,
                                           model.pvalues):
            results.append({
                'T1': t1,
                'T2': t2,
                'Subject': sub,
                'Regressor': param,
                'Coefficient': coef,
                'T-value': tval,
                'P-value': pval
            })
    except Exception as e:
        # In the error case, we fill in NaNs for the expected regressors.
            results.append({
                'T1': t1,
                'T2': t2,
                'Subject': sub,
                'Regressor': "None",
                'Coefficient': float('nan'),
                'T-value': float('nan'),
                'P-value': float('nan')
            })
            
    return results


########################
# Run main effect model
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat "
all_results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, model_syntax = model_syntax ) for t1, t2 in product(range(0,num_tps,10), range(0,num_tps,10)) )
# Flatten the list of lists to get one flat list of dictionaries.
final_results = [item for sublist in all_results for item in sublist]
# (Optional) Convert to DataFrame if desired:
results_df = pd.DataFrame(final_results)
print(results_df.head())
results_df.to_csv('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_main_effect_model_regression_results.csv', index=False)

########################
# Model 1.1, cue and task switch effect?
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS:Task_repeat + IDS:Task_switch + EDS:Task_repeat + EDS:Task_switch + Response_repeat"
all_results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, model_syntax = model_syntax ) for t1, t2 in product(range(0,num_tps,10), range(0,num_tps,10)) )
# Flatten the list of lists to get one flat list of dictionaries.
final_results = [item for sublist in all_results for item in sublist]
# (Optional) Convert to DataFrame if desired:
results_df = pd.DataFrame(final_results)
print(results_df.head())
results_df.to_csv('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_switch_model_regression_results.csv', index=False)


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
    "EDS:Task_switch:Response_switch ")
all_results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, model_syntax = model_syntax ) for t1, t2 in product(range(0,num_tps,10), range(0,num_tps,10)) )
# Flatten the list of lists to get one flat list of dictionaries.
final_results = [item for sublist in all_results for item in sublist]
# (Optional) Convert to DataFrame if desired:
results_df = pd.DataFrame(final_results)
print(results_df.head())
results_df.to_csv('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_response_switch_model_regression_results.csv', index=False)

########################
# Model 2, understand the effect of accuracy
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Stay:Accu + IDS:Accu + EDS:Accu + Stay:Cue_repeat:Accu + Stay:Cue_switch:Accu"
all_results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, model_syntax = model_syntax ) for t1, t2 in product(range(0,num_tps,10), range(0,num_tps,10)) )
# Flatten the list of lists to get one flat list of dictionaries.
final_results = [item for sublist in all_results for item in sublist]
# (Optional) Convert to DataFrame if desired:
results_df = pd.DataFrame(final_results)
print(results_df.head())
results_df.to_csv('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_accuracy_model_regression_results.csv', index=False)


########################
# Model 3, understand the effect of RT
########################
model_syntax = "ds ~ 0 + Stay:Cue_repeat + Stay:Cue_switch + IDS + EDS + Response_repeat + Task_repeat + Stay:Cue_repeat:RT + Stay:Cue_switch:RT + IDS:RT + EDS:RT"
all_results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, model_syntax = model_syntax ) for t1, t2 in product(range(0,num_tps,10), range(0,num_tps,10)) )
# Flatten the list of lists to get one flat list of dictionaries.
final_results = [item for sublist in all_results for item in sublist]
# (Optional) Convert to DataFrame if desired:
results_df = pd.DataFrame(final_results)
print(results_df.head())
results_df.to_csv('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_RT_model_regression_results.csv', index=False)






########################
# # Function to perform second-level statistics

# def group_level_stats_flexible(results_df, param1, param2=None, roisize=418):
#     """
#     Perform group-level statistics (one-sample t-test or paired t-test) for given regressors.

#     Parameters:
#     - results_df: DataFrame containing regression results.
#     - param1: Name of the first regressor to analyze (always used).
#     - param2: Name of the second regressor for comparison (optional).
#     - roisize: Number of ROIs in the template.

#     Returns:
#     - group_coefficients: Array of group-level mean coefficients for param1 (if param2 is None).
#     - group_t_values: Array of group-level t-values (one-sample or paired, based on param2).
#     """
#     group_coefficients = np.zeros(roisize)
#     group_t_values = np.zeros(roisize)

#     for roi in range(roisize):
#         if param2 is None:
#             # One-sample t-test for param1 against 0
#             roi_coefficients = results_df.loc[
#                 (results_df['ROI'] == roi) & (results_df['Regressor'] == param1), 'Coefficient'
#             ].dropna()

#             if not roi_coefficients.empty:
#                 t_stat, _ = ttest_1samp(roi_coefficients, popmean=0, nan_policy='omit')
#                 group_coefficients[roi] = roi_coefficients.mean()
#                 group_t_values[roi] = t_stat

#         else:
#             # Paired t-test between param1 and param2
#             roi_coeff1 = results_df.loc[
#                 (results_df['ROI'] == roi) & (results_df['Regressor'] == param1), 'Coefficient'
#             ].dropna()
#             roi_coeff2 = results_df.loc[
#                 (results_df['ROI'] == roi) & (results_df['Regressor'] == param2), 'Coefficient'
#             ].dropna()

#             if not roi_coeff1.empty and not roi_coeff2.empty and len(roi_coeff1) == len(roi_coeff2):
#                 t_stat, _ = ttest_rel(roi_coeff1, roi_coeff2)
#                 group_coefficients[roi] = roi_coeff1.mean() - roi_coeff2.mean()  # Mean difference
#                 group_t_values[roi] = t_stat

#     return group_coefficients, group_t_values