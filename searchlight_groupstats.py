# do group level stats on searchlight results
import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from nilearn.masking import apply_mask, unmask
from tqdm import tqdm

# ------------- Config -----------------
regressor_name = "EDS_vs_IDS" # Name of the regressor for the group stats
beta_suffix = f"{regressor_name}_beta.nii.gz"

data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA_regression"
mask_path = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz"
output_dir = os.path.join(data_dir, "group_stats")
#os.makedirs(output_dir, exist_ok=True)

mask_img = nib.load(mask_path)

# ------------- Load All Subject β-Maps -----------------
subject_beta_files = sorted(glob(os.path.join(data_dir, f"*_{beta_suffix}")))
subject_ids = [os.path.basename(f).split("_")[0] for f in subject_beta_files]
n_subjects = len(subject_beta_files)

print(f"Found {n_subjects} subjects for regressor {regressor_name}")

# Load all beta maps into matrix of shape (n_subjects, n_voxels)
all_data = np.array([
    apply_mask(nib.load(f), mask_img) for f in tqdm(subject_beta_files, desc="Loading subjects")
])  # shape: (n_subjects, n_voxels)

# ------------- Parallel T-test Across Voxels -----------------
def ttest_voxel(v_idx):
    y = all_data[:, v_idx]
    if np.all(np.isnan(y)):
        return np.nan, np.nan
    tval, pval = ttest_1samp(y[~np.isnan(y)], popmean=0)
    return tval, pval

print("Running voxelwise t-tests...")
results = Parallel(n_jobs=16)(
    delayed(ttest_voxel)(v) for v in tqdm(range(all_data.shape[1]), desc="T-tests")
)

# Extract results
tvals = np.array([r[0] for r in results])
pvals = np.array([r[1] for r in results])

# Optional: FDR correction
from statsmodels.stats.multitest import fdrcorrection
fdr_mask, fdr_pvals = fdrcorrection(pvals, alpha=0.05)

# Unmask and save
group_beta = np.nanmean(all_data, axis=0)
group_beta_img = unmask(group_beta, mask_img)
group_beta_img.to_filename(os.path.join(output_dir, f"group_{regressor_name}_beta_mean.nii.gz"))

group_tval_img = unmask(tvals, mask_img)
group_tval_img.to_filename(os.path.join(output_dir, f"group_{regressor_name}_tval.nii.gz"))

group_pval_img = unmask(pvals, mask_img)
group_pval_img.to_filename(os.path.join(output_dir, f"group_{regressor_name}_pval.nii.gz"))

group_fdr_img = unmask(fdr_mask.astype(np.float32), mask_img)
group_fdr_img.to_filename(os.path.join(output_dir, f"group_{regressor_name}_fdr_mask.nii.gz"))

print(f"✓ Finished group stats for {regressor_name}")
