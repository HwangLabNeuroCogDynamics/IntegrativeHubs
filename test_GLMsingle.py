#testing GLMSingle
# to install, on thalamege, do pip install git+https://github.com/cvnlab/GLMsingle.git
# to install on argon, first "source activate" into your env, then do "pip install git+https://github.com/cvnlab/GLMsingle.git"
import numpy as np
import os
from os.path import join, exists, split
import time
from glmsingle.glmsingle import GLM_single
from nilearn.input_data import NiftiMasker
import statsmodels.formula.api as smf
import pandas as pd
from joblib import Parallel, delayed
import pandas as pd
import subprocess
import re

def voxelwise_glm( Y, design_df, formula, masker: NiftiMasker, out_dir: str, prefix: str, n_jobs: int = 24 ):
    """
    Mass‐univariate OLS per voxel, parallelized.

    Y           : (n_trials, n_voxels) array of trial‐wise data
    design_df   : DataFrame with n_trials rows, columns for your regressors
    formula     : use 'y' as the dependent var, e.g. 'y ~ RT_z + Trial_type'
    masker      : fitted NiftiMasker
    out_dir     : where to write your maps
    prefix      : filename prefix for each map, here we should use the subject ID
    n_jobs      : number of parallel jobs (24)
    """

    # ensure out_dir exists
    #os.makedirs(out_dir, exist_ok=True)

    n_trials, n_voxels = Y.shape

    # --- 1) do a single fit for voxel 0 to seed term names & array shapes ---
    df0 = design_df.copy()
    df0['y'] = Y[:, 0]
    m0  = smf.ols(formula, data=df0).fit()
    term_names = m0.params.index.tolist()   # e.g. ['Intercept','RT_z','Trial_type[T.EDS]',...]
    p = len(term_names)

    # preallocate storage
    betas  = np.zeros((p, n_voxels), dtype=float)
    tstats = np.zeros((p, n_voxels), dtype=float)

    # fill in voxel 0
    betas[:, 0]  = m0.params.values
    tstats[:, 0] = m0.tvalues.values

    # --- 2) define worker for a single voxel ---
    def _fit_voxel(v):
        df = design_df.copy()
        df['y'] = Y[:, v]
        m  = smf.ols(formula, data=df).fit()
        return m.params.values, m.tvalues.values

    # --- 3) parallel loop over remaining voxels ---
    results = Parallel(n_jobs=n_jobs, verbose=5)( delayed(_fit_voxel)(v) for v in range(1, n_voxels) )

    # unpack results
    for idx, (b_vec, t_vec) in enumerate(results, start=1):
        betas[:, idx]  = b_vec
        tstats[:, idx] = t_vec

    # --- 4) inverse‐transform & save maps ---
    print(term_names)
    for i, name in enumerate(term_names):
        safe_name = make_safe(name)
        # beta map
        img_b = masker.inverse_transform(betas[i, :])
        fout_b = os.path.join(out_dir, f"{prefix}_{safe_name}_beta.nii.gz")
        img_b.to_filename(fout_b)
        print("Wrote", fout_b)

        # t‐stat map
        img_t = masker.inverse_transform(tstats[i, :])
        fout_t = os.path.join(out_dir, f"{prefix}_{safe_name}_tstat.nii.gz")
        img_t.to_filename(fout_t)
        print("Wrote", fout_t)

    return {'betas': betas, 'tstats': tstats, 'terms': term_names}

def make_safe(s):
    # replace runs of non‐word characters with a single underscore
    return re.sub(r"[^\w]+", "_", s).strip("_")

# load vars
#subjects = pd.read_csv( "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv" )['sub']
subjects = input()
n_runs = 8
TR = 1.8
stim_shift = 0.6  #(6s delay for thalhi per Dillan's code − 3*1.8s dropped = 0.6s)

for sub in [subjects]:
    # define paths
    deconv_dir = f"/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-{sub}"
    data_dir   = f"/Shared/lss_kahwang_hpc/data/ThalHi/fmriprep/sub-{sub}/func"
    mask_afni  = os.path.join(deconv_dir, "combined_mask+tlrc")
    mask_nii   = os.path.join(deconv_dir, "combined_mask.nii.gz")
    stim_file  = os.path.join(deconv_dir, "cue_stimtimes.1D")
    onsets_mat = np.loadtxt(stim_file)
    
    # subject‐specific GLMsingle output
    out_base = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle"
    outputdir = os.path.join(out_base, f"sub-{sub}")
    
    try:
        # make sure output folder exists
        #os.makedirs(outputdir, exist_ok=True)

        # convert mask to NIfTI and fit masker once. This is because oroginal overlap mask is in AFNI format, and GLMsingle only accepts NIfTI.
        #subprocess.run( ["3dAFNItoNIFTI", "-prefix", mask_nii, mask_afni], check=True )
        masker = NiftiMasker(mask_img=mask_nii, standardize=False, detrend=False)
        masker.fit()

        # 2) build data & design lists per GLMsingle
        data = []
        design = []
        for run in range(n_runs):
            # load & mask BOLD, then drop first 3 TRs
            fn = f"sub-{sub}_task-ThalHi_run-{run+1}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            bold = os.path.join(data_dir, fn)
            ts = masker.transform(bold).T
            ts = ts[:, 3:] # drop first 3 TRs
            data.append(ts)
            
            # build onsets vector for that run
            n_tp = ts.shape[1] # should be 213
            on = np.zeros(n_tp)
            tmp_mat = onsets_mat[run][~np.isnan(onsets_mat[run])]
            idx = np.round(tmp_mat / TR).astype(int) #round to nearest TR... see https://glmsingle.readthedocs.io/en/latest/wiki.html#my-experiment-design-is-not-quite-synchronized-with-my-fmri-data
            on[idx] = 1
            design.append(on[:, np.newaxis])

        # 3) run GLMsingle
        opt = {
            'wantlibrary': 1,
            'wantglmdenoise':1,
            'wantfracridge': 1,
            'wantfileoutputs': [0,0,0,0],
            'wantmemoryoutputs':[1,1,1,1]
        }
        glms = GLM_single(opt)
        start = time.time()
        results_glmsingle = glms.fit(design, data, stimdur=0.5, tr=TR, outputdir=outputdir)
        print(f"sub-{sub} GLMsingle done in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")

        # 4) write out GLMsingle maps
        # R2, noise pool, HRF index, meanBeta, TrialBetas
        typed = results_glmsingle['typed']
        for key in ['R2', 'noisepool', 'HRFindex']:
            vol = masker.inverse_transform(np.squeeze(typed[key]))
            vol.to_filename(os.path.join(outputdir, f"{sub}_{key}.nii.gz"))
        # mean beta
        meanb = np.squeeze(typed['betasmd']).mean(axis=1)
        masker.inverse_transform(meanb).to_filename(os.path.join(outputdir, f"{sub}_meanBeta.nii.gz"))
        # trial betas. # this is the LSS for later use.
        tb = np.squeeze(typed['betasmd']).T
        masker.inverse_transform(tb).to_filename(os.path.join(outputdir, f"{sub}_TrialBetas.nii.gz"))

        # 5) run voxelwise GLM on the single‐trial betas as validation
        # Y = tb   # (n_trials, n_voxels)
        # behav = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
        # subdf = behav[behav['sub']==sub].reset_index(drop=True)
        # for c in ["Trial_type","Task"]:
        #     subdf[c] = subdf[c].astype("category")
        # voxelwise_glm( Y, subdf, formula="y ~ C(Trial_type, Treatment(reference='Stay'))", masker=masker, out_dir=outputdir, prefix=str(sub), n_jobs=24 )

    except Exception as e:
        # write out a warning file if anything fails
        warn_file = os.path.join(outputdir, f"{sub}_warning.txt")
        with open(warn_file, "w") as f:
            f.write(f"Error processing sub-{sub}:\n{str(e)}\n")
        print(f"[!] sub-{sub} failed; see {warn_file}")
        # then continue to next subject
        continue




# graveyard

### use AFNI 3dDeconvolve to create the design matrix
# note this part is calling on AFNI in python, this is not a python function.
# it is easy to do this on thalamege, but on argon AFNI needs to be run in container so I am not sure how to do this.
# one solution is to just write a bash script and run this part separately, should be easy and fast to do.
# Note here not reating different cue types as different conditions, don't think its necessary when the goal is to get the GLM coefficients for each trial.  
# xmat_file = os.path.join(deconv_dir, "cueonset_IM.xmat.1D")
#concat_str = '1D: ' + ' '.join(str(i) for i in run_breaks[0:8])
# cmd = [
#     "3dDeconvolve",
#       "-nodata", str(n_timepts), str(TR),
#       "-polort", "-1",
#       "-concat", concat_str,
#       "-num_stimts", "1",
#       "-local_times",
#       "-stim_times", "1",
#         os.path.join(deconv_dir, "cue_stimtimes.1D"),
#         "TENT(0.6,2.4,2)",
#       "-stim_label", "1", "cue",
#       "-x1D", xmat_file,
#       "-x1D_stop",
#       "-allzero_OK"
# ]
# print("Running 3dDeconvolve:"," ".join(cmd))
# subprocess.run(cmd, check=True)

# load the full design‐matrix, extract first "tent", binarize, and save
#mat = np.loadtxt(xmat_file)      # comments auto‐ignored
#onsets = ( mat[:, 0] != 0).astype(int)   # convert on set to binary, will loose some termporal percision without AFNI's internal up then down sampling of full design matrix.
# see: https://glmsingle.readthedocs.io/en/latest/wiki.html#my-experiment-design-is-not-quite-synchronized-with-my-fmri-data 