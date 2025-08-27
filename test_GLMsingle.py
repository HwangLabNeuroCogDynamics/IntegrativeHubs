####################################################################
# Script to run GLMsingle on the ThalHi dataset
####################################################################
# -----------------------------------------------------------------
# References & resources
# -----------------------------------------------------------------
# Install GLMsingle:
#   On thalamege: pip install --user git+https://github.com/cvnlab/GLMsingle.git
#   On argon:     source activate <env> && pip install git+https://github.com/cvnlab/GLMsingle.git
#
# Paper: https://elifesciences.org/articles/77599
# Video: https://www.youtube.com/watch?v=xFO-cYHZEwg
# Docs:  https://glmsingle.readthedocs.io/en/latest/python.html
# -----------------------------------------------------------------

import numpy as np
import os
from os.path import join, exists, split
import time
from glmsingle.glmsingle import GLM_single
from nilearn.input_data import NiftiMasker
import statsmodels.formula.api as smf
import pandas as pd
from joblib import Parallel, delayed
import subprocess
import re

# --------------------- CONFIG ---------------------
n_runs = 8
TR = 1.8
stim_shift = 0.6   # 6s delay per Dillan’s code − (3 * 1.8s dropped) = 0.6s
# subjects = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
subjects = input()  # hardcode subject ID here for testing
# -------------------------------------------------


# ======== Voxelwise GLM helpers ========
def voxelwise_glm(Y, design_df, formula, masker: NiftiMasker, out_dir: str, prefix: str, n_jobs: int = 24):
    """
    Mass-univariate OLS per voxel, parallelized.

    Y         : (n_trials, n_voxels) trial-wise array
    design_df : DataFrame with n_trials rows and regressors
    formula   : use 'y' as dependent var, e.g. 'y ~ RT_z + Trial_type'
    masker    : fitted NiftiMasker
    out_dir   : where to write maps
    prefix    : filename prefix (subject ID recommended)
    n_jobs    : number of parallel jobs (default=24)
    """

    n_trials, n_voxels = Y.shape

    # Seed term names and array shapes with voxel 0
    df0 = design_df.copy()
    df0['y'] = Y[:, 0]
    m0 = smf.ols(formula, data=df0).fit()
    term_names = m0.params.index.tolist()
    p = len(term_names)

    # Preallocate storage
    betas = np.zeros((p, n_voxels), dtype=float)
    tstats = np.zeros((p, n_voxels), dtype=float)

    # Fill voxel 0
    betas[:, 0] = m0.params.values
    tstats[:, 0] = m0.tvalues.values

    # Worker for a single voxel
    def _fit_voxel(v):
        df = design_df.copy()
        df['y'] = Y[:, v]
        m = smf.ols(formula, data=df).fit()
        return m.params.values, m.tvalues.values

    # Parallel loop over remaining voxels
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_voxel)(v) for v in range(1, n_voxels)
    )

    # Unpack results
    for idx, (b_vec, t_vec) in enumerate(results, start=1):
        betas[:, idx] = b_vec
        tstats[:, idx] = t_vec

    # Inverse-transform & save maps
    print(term_names)
    for i, name in enumerate(term_names):
        safe_name = make_safe(name)

        # Beta map
        img_b = masker.inverse_transform(betas[i, :])
        fout_b = os.path.join(out_dir, f"{prefix}_{safe_name}_beta.nii.gz")
        img_b.to_filename(fout_b)
        print("Wrote", fout_b)

        # T-stat map
        img_t = masker.inverse_transform(tstats[i, :])
        fout_t = os.path.join(out_dir, f"{prefix}_{safe_name}_tstat.nii.gz")
        img_t.to_filename(fout_t)
        print("Wrote", fout_t)

    return {'betas': betas, 'tstats': tstats, 'terms': term_names}


def make_safe(s):
    """Replace runs of non-word chars with underscores."""
    return re.sub(r"[^\w]+", "_", s).strip("_")

# =================================================================
# Main subject loop
# =================================================================
for sub in [subjects]:
    # Define paths
    deconv_dir = f"/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{sub}"
    data_dir = f"/Shared/lss_kahwang_hpc/data/ThalHi/fmriprep/sub-{sub}/func"
    mask_afni = os.path.join(deconv_dir, "combined_mask+tlrc")
    mask_nii = os.path.join(deconv_dir, "combined_mask.nii.gz")
    stim_file = os.path.join(deconv_dir, "cue_stimtimes.1D")
    onsets_mat = np.loadtxt(stim_file)

    # Subject-specific GLMsingle output
    out_base = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle"
    outputdir = os.path.join(out_base, f"sub-{sub}")

    try:
        # Convert AFNI mask to NIfTI and fit masker
        # subprocess.run(["3dAFNItoNIFTI", "-prefix", mask_nii, mask_afni], check=True)
        masker = NiftiMasker(mask_img=mask_nii, standardize=False, detrend=False)
        masker.fit()

        # Build data & design lists per GLMsingle
        data = []
        design = []
        for run in range(n_runs):
            # Load & mask BOLD, then drop first 3 TRs
            fn = f"sub-{sub}_task-ThalHi_run-{run+1}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            bold = os.path.join(data_dir, fn)
            ts = masker.transform(bold).T
            ts = ts[:, 3:]  # drop first 3 TRs
            data.append(ts)

            # Build onsets vector for that run
            n_tp = ts.shape[1]
            on = np.zeros(n_tp)
            tmp_mat = onsets_mat[run][~np.isnan(onsets_mat[run])]
            idx = np.round(tmp_mat / TR).astype(int)  # round to nearest TR
            on[idx] = 1
            design.append(on[:, np.newaxis])

        # Run GLMsingle
        opt = {
            'wantlibrary': 1,
            'wantglmdenoise': 1,
            'wantfracridge': 1,
            'wantfileoutputs': [0, 0, 0, 0],
            'wantmemoryoutputs': [1, 1, 1, 1]
        }
        glms = GLM_single(opt)
        start = time.time()
        results_glmsingle = glms.fit(design, data, stimdur=0.5, tr=TR, outputdir=outputdir)
        print(f"sub-{sub} GLMsingle done in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")

        # Write out GLMsingle maps
        typed = results_glmsingle['typed']

        # R2, noise pool, HRF index
        for key in ['R2', 'noisepool', 'HRFindex']:
            vol = masker.inverse_transform(np.squeeze(typed[key]))
            vol.to_filename(os.path.join(outputdir, f"{sub}_{key}.nii.gz"))

        # Mean beta
        meanb = np.squeeze(typed['betasmd']).mean(axis=1)
        masker.inverse_transform(meanb).to_filename(os.path.join(outputdir, f"{sub}_meanBeta.nii.gz"))

        # Trial betas (LSS, later use)
        tb = np.squeeze(typed['betasmd']).T
        masker.inverse_transform(tb).to_filename(os.path.join(outputdir, f"{sub}_TrialBetas.nii.gz"))

        # Voxelwise GLM validation (commented example)
        # Y = tb
        # behav = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
        # subdf = behav[behav['sub']==sub].reset_index(drop=True)
        # for c in ["Trial_type","Task"]:
        #     subdf[c] = subdf[c].astype("category")
        # voxelwise_glm(Y, subdf, formula="y ~ C(Trial_type, Treatment(reference='Stay'))",
        #               masker=masker, out_dir=outputdir, prefix=str(sub), n_jobs=24)

    except Exception as e:
        # On failure, write out a warning file
        warn_file = os.path.join(outputdir, f"{sub}_warning.txt")
        with open(warn_file, "w") as f:
            f.write(f"Error processing sub-{sub}:\n{str(e)}\n")
        print(f"[!] sub-{sub} failed; see {warn_file}")
        continue


#end of line
