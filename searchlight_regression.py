# do distance regression for each subject, fit model for each searchlight sphere
import os
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from nilearn.masking import unmask


# ------------------------ Config ------------------------
data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA_regression"
os.makedirs(out_dir, exist_ok=True)

mask_img_path = os.path.join(data_dir, "searchlight_mask.nii.gz")
mask_nii = nib.load(mask_img_path)
mask_data = mask_nii.get_fdata()
sphere_inds = np.where(mask_data > 0)

behavior_csv = "/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_zRTs_full.csv"
df = pd.read_csv(behavior_csv)

model_syntax = (
    "ds ~ zRT*C(Trial_type, Treatment(reference='Stay')) + perceptual_change + "
    "C(response_repeat) + C(task_repeat) + "
    "C(prev_target_feature_match , Treatment(reference='switch_target_feature'))"
)

# ------------------------ Spherewise Regression ------------------------
def regress_one_sphere(sphere_index, sdf, ds_array):
    sdf = sdf.copy()
    sdf["ds"] = ds_array[sphere_index, :]
    beta = {}
    tval = {}

    try:
        model = smf.ols(model_syntax, data=sdf).fit()
        # extract all standard fixed effects
        beta = model.params.to_dict()
        tval = model.tvalues.to_dict()

        # === EDS vs IDS contrast ===
        fe_names = model.params.index.tolist()
        fe_vals  = model.params.values
        k_fe     = len(fe_names)

        try:
            pos_eds = fe_names.index("C(Trial_type, Treatment(reference='Stay'))[T.EDS]")
            pos_ids = fe_names.index("C(Trial_type, Treatment(reference='Stay'))[T.IDS]")
        except ValueError:
            pass  # one or both regressors not present
        else:
            L = np.zeros((1, k_fe))
            L[0, pos_eds] = 1
            L[0, pos_ids] = -1
            ct = model.t_test(L)

            beta["EDS_vs_IDS"] = fe_vals[pos_eds] - fe_vals[pos_ids]
            tval["EDS_vs_IDS"] = np.squeeze(ct.tvalue)

    except Exception as e:
        # leave beta and tval as empty dicts (will fill with NaNs later)
        pass

    return beta, tval

# ------------------------ Process One Subject ------------------------
def process_subject(sub_id):
    print(f"→ Processing subject {sub_id}")
    subj_file = os.path.join(data_dir, f"{sub_id}_sl_adjcorr.npy")
    if not os.path.exists(subj_file):
        print(f"✗ Missing data for {sub_id}")
        return

    ds_array = np.load(subj_file)  # shape (n_spheres, n_trials)
    n_spheres = ds_array.shape[0]

    # Load and prepare behavioral data
    sdf = df[df["sub"] == int(sub_id)].copy().sort_values(["block", "trial_n"]).reset_index(drop=True)
    sdf = sdf.assign(response_repeat=(sdf['Subject_Respo'] == sdf['prev_resp']).astype("category"),
                     task_repeat=(sdf['Task'] == sdf['prev_task']).astype("category"))
    for col in ["Trial_type", "prev_target_feature_match"]:
        sdf[col] = sdf[col].astype("category")
    sdf = sdf[(sdf['trial_n'] != 0) &
              (sdf['trial_Corr'] != 0) &
              (sdf['zRT'] <= 3) &
              (sdf['prev_accuracy'] != 0)].reset_index(drop=True)
    trial_vec = (sdf['block'] - 1) * 51 + sdf['trial_n'] #figure out the trial idx after dropping trials
    ds_array = ds_array[:, trial_vec] #drop trials in the ds array

    # Regression per sphere
    results = Parallel(n_jobs=16, verbose = 5)(delayed(regress_one_sphere)(i, sdf, ds_array) for i in range(n_spheres))

    # Collect regressor names
    reg_names = sorted(set(k for beta, _ in results for k in beta.keys()))

    # output arrays (n_regressors × n_spheres)
    betas = {r: np.full(n_spheres, np.nan) for r in reg_names}
    tvals = {r: np.full(n_spheres, np.nan) for r in reg_names}
    for i, (beta, tval) in enumerate(results):
        for r in reg_names:
            betas[r][i] = beta.get(r, np.nan)
            tvals[r][i] = tval.get(r, np.nan)

    # Save to nii using masking.unmask
    for r in reg_names:
        r_clean = r.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_").replace("/", "_")
        beta_img = unmask(betas[r], mask_nii)
        tval_img = unmask(tvals[r], mask_nii)

        beta_img.to_filename(os.path.join(out_dir, f"{sub_id}_{r_clean}_beta.nii.gz"))
        tval_img.to_filename(os.path.join(out_dir, f"{sub_id}_{r_clean}_tval.nii.gz"))

    print(f"✓ Finished subject {sub_id}")

# ------------------------ Run All Subjects ------------------------
if __name__ == "__main__":
    #all_files = [f for f in os.listdir(data_dir) if f.endswith("_sl_adjcorr.npy")]
    #subjects = sorted(set(f.split("_")[0] for f in all_files))
    subjects = input()

    for sub in [subjects]:
        process_subject(sub)
