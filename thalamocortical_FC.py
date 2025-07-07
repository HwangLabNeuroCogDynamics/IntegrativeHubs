##########################################
#trial by trial thalamocortical connectivity 
##########################################

import nibabel as nib
from nilearn.maskers import NiftiMasker
import numpy as np
import os
from os.path import join, exists, split
from joblib import Parallel, delayed
import pandas as pd
import statsmodels.formula.api as smf
from numpy.linalg import inv

#for argon submission 
sub = input()
sub = str(sub)

dataset_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/"
behavior_csv = "/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_zRTs_full2.csv"
df = pd.read_csv(behavior_csv)


# this is the vectorized version of the PPI function above, which is MUCH FASTER.
def compute_ppi_vec(i, param, Th_ts, Ctx_ts):
    """
    Vectorized PPI for thalamus voxel i:
      - Builds design X = [1, Th_i, Th_i * param] # include the interaction of parametric modulation
      - Fits B = (X'X)^{-1} X' Y  where Y = Ctx_ts (T x n_ctx)
      - Computes t-values for each coefficient across all n_ctx voxels

    Returns four 1D arrays (length n_ctx):
      beta_main, beta_int,  t_main,  t_int
    """
    T, n_ctx = Ctx_ts.shape

    # 1) build design matrix X (T×3)
    th = Th_ts[:, i]
    X  = np.column_stack([
        np.ones(T, dtype=np.float32), # this is intercept
        th.astype(np.float32), # this is the thalamus time series
        (th * param).astype(np.float32) # this is the interaction term between thalamus and param
    ])  # shape (T,3)

    # 2) precompute (X'X)^{-1} and pseudoinverse
    XtX_inv = inv(X.T @ X) # the @ means matrix multiplication
    pinv    = XtX_inv @ X.T            

    # 3) estimate betas for all cortex voxels at once
    #    B has shape (3, n_ctx)
    B = pinv @ Ctx_ts # estimate the coefficients for each cortex voxel                 

    # 4) residuals and variance per voxel
    Yhat = X @ B # what is the predicted value for each cortex voxel                      
    R    = Ctx_ts - Yhat  # what is the residual, difference between actual and predicted             
    SSE  = (R**2).sum(axis=0) #sum of squared errors for each cortex voxel (1D array of length n_ctx)         
    df   = T - 3 # what is the DOF given 3 predicctors (intercept, Thalamus, interaction) and T time points
    sigma2 = SSE / df  # variance of the residuals for each cortex voxel (1D array of length n_ctx)                

    # 5) standard errors for each coef and each voxel
    #    var(B_k) = diag(XtX_inv)[k] * sigma2
    se_coef = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))  # standard error for each coefficient (3 x n_ctx)

    # 6) t-values: B / se
    se_coef[se_coef == 0] = np.nan  # avoid div/0 # get rid of zeros in standard errors to avoid division by zero
    with np.errstate(invalid='ignore'): # ignore warnings about division by zero
        Tvals = B / se_coef # t-values for each coefficient given standard errors (3 x n_ctx)

    # return main (k=1) and interaction (k=2). Not the first (k=0) is intercept which we dont need
    return B[1, :], B[2, :], Tvals[1, :], Tvals[2, :]


# load the trial by trial betas from GLMsingle
trialbetas = nib.load(os.path.join(dataset_dir, f"GLMsingle/sub-{sub}/{sub}_TrialBetas.nii.gz"))

# define the masker
thalamus_masker = NiftiMasker(os.path.join("/Shared/lss_kahwang_hpc/data/TRIIMS/", "GLMsingle/ROIs/thmask.nii.gz"))
ctx_masker = NiftiMasker(os.path.join("/Shared/lss_kahwang_hpc/data/TRIIMS/", "GLMsingle/ROIs/searchlight_mask.nii.gz"))

Th_ts = thalamus_masker.fit_transform(trialbetas)
Ctx_ts = ctx_masker.fit_transform(trialbetas)
n_trials = Th_ts.shape[0] #how many trials
tha_voxels = Th_ts.shape[1] #how many thalamus voxels
ctx_voxels = Ctx_ts.shape[1] #how many cortex voxels
n_jobs = 24  # this means we will run 24 loops at a same time.

## this is the faster version.
sub_df = df[df["sub"] == int(sub)].sort_values(["block", "trial_n"]).reset_index(drop=True)
#check if the number of trials match
if n_trials != len(sub_df):
    #need to create a trial selection vector to figure out which trials got dropped
    trial_vec = (sub_df['block'] - 1) * 51 + sub_df['trial_n']
    Th_ts = Th_ts[trial_vec, :]
    Ctx_ts = Ctx_ts[trial_vec, :]

param = 1.0*(sub_df['Trial_type']=="EDS")
# parallel compute each thalamus‐to‐cortex with regression
results = Parallel(n_jobs=24, verbose=5)(
    delayed(compute_ppi_vec)(i, param, Th_ts, Ctx_ts)
    for i in range(Th_ts.shape[1])
)

# results is a list of tuples (beta_main_row, beta_int_row, t_main_row, t_int_row)
#Stack them into full matrices:
beta_main_rows, beta_int_rows, t_main_rows, t_int_rows = zip(*results)

#Stack each sequence of 1×n_ctx rows into a full matrix (n_th × n_ctx)
beta_main_mat = np.vstack(beta_main_rows)
beta_int_mat  = np.vstack(beta_int_rows)
t_main_mat    = np.vstack(t_main_rows)
t_int_mat     = np.vstack(t_int_rows)

# save the results
np.save(os.path.join(dataset_dir, f"GLMsingle/sub-{sub}/thalamus_ctx_ppi_main_beta.npy"), beta_main_mat)
np.save(os.path.join(dataset_dir, f"GLMsingle/sub-{sub}/thalamus_ctx_ppi_interaction_beta.npy"), beta_int_mat)
np.save(os.path.join(dataset_dir, f"GLMsingle/sub-{sub}/thalamus_ctx_ppi_main_t.npy"), t_main_mat)
np.save(os.path.join(dataset_dir, f"GLMsingle/sub-{sub}/thalamus_ctx_ppi_interaction_t.npy"), t_int_mat)



