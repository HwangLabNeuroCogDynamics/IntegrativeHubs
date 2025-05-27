# do distance regression for each subject, fit model for each searchlight sphere
import os
### this is necessary when using joblib to avoid parallelization issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from nilearn.masking import unmask
import re

def clean_regressor_name(reg_name):
    clean = reg_name
    # simplify any remaining C(...) contrasts
    clean = re.sub(r"C\(([^,]+),\s*Treatment\(reference=['\"]?([^'\"]+)['\"]?\)\)\[T\.([^\]]+)\]", r"\1_vs_\3", clean)
    clean = re.sub(r"C\(([^)]+)\)\[T\.([^\]]+)\]", r"\1_vs_\2", clean)
    # replace non-alphanumeric with underscore
    clean = re.sub(r"[^\w]+", "_", clean)
    # collapse multiple underscores
    clean = re.sub(r"_+", "_", clean)
    # strip leading/trailing underscores
    clean = clean.strip("_")

    return clean

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

# ------------------------ Spherewise Regression ------------------------
def regress_one_sphere(sphere_index, sdf, ds_array, model_syntax):
    """
    Fit OLS for one searchlight sphere, extract betas/t-values,
    and compute both overall and condition-specific marginal effects, had to do that one by one...
    """
    sdf = sdf.copy()
    sdf["ds"] = ds_array[sphere_index, :]
    beta = {}
    tval = {}

    try:
        model = smf.ols(model_syntax, data=sdf).fit()
        params = model.params

        # Store raw betas and t-values
        beta.update(params.to_dict())
        tval.update(model.tvalues.to_dict())

        # Prepare trial frequencies for weighting marginal effects
        # Trial_type frequencies
        freq_tt   = sdf['Trial_type'].value_counts(normalize=True)
        p_IDS     = float(freq_tt.get('IDS', 0))
        p_EDS     = float(freq_tt.get('EDS', 0))
        p_Stay    = float(freq_tt.get('Stay', 0))

        #response_repeat frequencies
        freq_rr   = sdf['response_repeat'].value_counts(normalize=True)
        p_resp    = float(freq_rr.get(True, 0))

        #task_repeat frequencies
        freq_tr   = sdf['task_repeat'].value_counts(normalize=True)
        p_task    = float(freq_tr.get(True, 0))

        #prev_target_feature_match frequencies
        freq_prev = sdf['prev_target_feature_match'].value_counts(normalize=True)
        p_prev    = float(freq_prev.get('same_target_feature', 0))

        # variable names...
        name_pc      = "perceptual_change_c"
        name_eds_pc  = "C(Trial_type, Treatment(reference='IDS'))[T.EDS]:perceptual_change_c"
        name_stay_pc = "C(Trial_type, Treatment(reference='IDS'))[T.Stay]:perceptual_change_c"
        name_rr   = "C(response_repeat)[T.True]"
        name_rrtr = f"{name_rr}:C(task_repeat)[T.True]"
        name_tr     = "C(task_repeat)[T.True]"
        name_prev   = [n for n in params.index if "prev_target_feature_match" in n and "]" in n][0]
        name_trprev = f"{name_tr}:{name_prev}"
        name_eds_main  = "C(Trial_type, Treatment(reference='IDS'))[T.EDS]"
        name_stay_main = "C(Trial_type, Treatment(reference='IDS'))[T.Stay]"

        # add maringal effect and tstat via linear contrast 
        def add_mfx(key, L):
            ct = model.t_test(L)
            beta[key] = float(ct.effect)
            tval[key] = float(ct.tvalue)

        #Overall  effect of perceptual_change_c (average over Trial_type)
        L_pc = [
            p_IDS  if n == name_pc      else
            p_EDS  if n == name_eds_pc  else
            p_Stay if n == name_stay_pc else
            0
            for n in params.index
        ]
        add_mfx('main_effect_perceptual_change_c', L_pc)

        # perceptual change slope in IDS
        L_ids = [1 if n == name_pc else 0 for n in params.index]
        add_mfx('perceptual_change_for_IDS', L_ids)

        # perceptual change slope in EDS = β_pc + β_eds_pc
        L_eds = [
            1 if n == name_pc     else
            1 if n == name_eds_pc else
            0
            for n in params.index
        ]
        add_mfx('perceptual_change_for_EDS', L_eds)

        # perceptual change slope inin Stay = β_pc + β_stay_pc
        L_stay = [
            1 if n == name_pc       else
            1 if n == name_stay_pc  else
            0
            for n in params.index
        ]
        add_mfx('perceptual_change_for_Stay', L_stay)

        # Main effect of response_repeat (avg over task_repeat)
        L_rr = [
            1      if n == name_rr   else
            p_task if n == name_rrtr else
            0
            for n in params.index
        ]
        add_mfx('main_effect_response_repeat', L_rr)

        # Main effect of task_repeat (avg over response_repeat & prev_target_feature_match)
        L_tr = [
            1      if n == name_tr      else
            p_resp if n == name_rrtr    else
            p_prev if n == name_trprev  else
            0
            for n in params.index
        ]
        add_mfx('main_effect_task_repeat', L_tr)

        # Main marginal effect of prev_target_feature_match (avg over task_repeat)
        L_prev = [
            1      if n == name_prev    else
            p_task if n == name_trprev  else
            0
            for n in params.index
        ]
        add_mfx('main_effect_prev_target_feature_match', L_prev)

        #Main effect: EDS vs IDS
        L_eds_main = [
            1 if n == name_eds_main else
            0
            for n in params.index
        ]
        add_mfx('main_effect_EDS_v_IDS', L_eds_main)

        #IDS vs Stay
        L_stay_main = [
            1 if n == name_stay_main else
            0
            for n in params.index
        ]
        add_mfx('main_effect_IDS_v_Stay', L_stay_main)

        # response_repeat × task_repeat
        L_rrtr_int = [
            1 if n == name_rrtr else
            0
            for n in params.index
        ]
        add_mfx('response_repeat_x_task_repeat', L_rrtr_int)

    except Exception:
        # If fitting or contrast fails, leave beta/tval dicts as-is
        pass

    return beta, tval


# ------------------------ Process One Subject ------------------------
def process_subject(sub_id, fn, model_syntax, model_tag):
    print(f"→ Processing subject {sub_id}")
    subj_file = os.path.join(data_dir, "%s_sl_adjcorr_%s.npy" %(sub_id, fn))
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
    sdf['perceptual_change_c'] = sdf['perceptual_change'] - sdf['perceptual_change'].mean() #center regresor
    trial_vec = (sdf['block'] - 1) * 51 + sdf['trial_n'] #figure out the trial idx after dropping trials
    ds_array = ds_array[:, trial_vec] #drop trials in the ds array

    # Regression per sphere
    results = Parallel(n_jobs=16, verbose = 5)(delayed(regress_one_sphere)(i, sdf, ds_array, model_syntax) for i in range(n_spheres))

    # Collect regressor names
    reg_names = sorted(set(k for beta, _ in results for k in beta.keys()))

    # output arrays (n_regressors × n_spheres)
    betas = {r: np.full(n_spheres, np.nan) for r in reg_names}
    tvals = {r: np.full(n_spheres, np.nan) for r in reg_names}
    for i, (beta, tval) in enumerate(results):
        for r in reg_names:
            betas[r][i] = beta.get(r, np.nan)
            tvals[r][i] = tval.get(r, np.nan)

    # Create subject-specific output directory
    subj_out_dir = os.path.join(out_dir, sub_id)
    os.makedirs(subj_out_dir, exist_ok=True)

    # Save to nii using masking.unmask
    for r in reg_names:
        r_clean = clean_regressor_name(r)
        beta_img = unmask(betas[r], mask_nii)
        tval_img = unmask(tvals[r], mask_nii)
        beta_img.to_filename(os.path.join(subj_out_dir, f"{r_clean}_{model_tag}_{fn}_beta.nii.gz"))
        tval_img.to_filename(os.path.join(subj_out_dir, f"{r_clean}_{model_tag}_{fn}_tval.nii.gz"))

    print(f"✓ Finished subject {sub_id} for {model_tag}")

# ------------------------ Run All Subjects ------------------------
if __name__ == "__main__":
    #all_files = [f for f in os.listdir(data_dir) if f.endswith("_sl_adjcorr.npy")]
    #subjects = sorted(set(f.split("_")[0] for f in all_files))
    subjects = input()

    for sub in [subjects]:
            # Define model(s)
        model1 = (
            "ds ~ zRT + C(Trial_type, Treatment(reference='IDS')) * perceptual_change_c + "
            "C(response_repeat) * C(task_repeat) + "
            "C(task_repeat) * C(prev_target_feature_match , Treatment(reference='switch_target_feature'))"
        )
        #process_subject(sub, fn = "WT", model_syntax=model1, model_tag="RTModel")
        process_subject(sub, fn = "resRTWTT", model_syntax=model1, model_tag="RTModel")
        #process_subject(sub, fn = "WF", model_syntax=model1, model_tag="RTModel")
        #process_subject(sub, fn = "resRTWF", model_syntax=model1, model_tag="RTModel")
        
        model2 = (
            "ds ~  C(Trial_type, Treatment(reference='IDS')) * perceptual_change_c + "
            "C(response_repeat) * C(task_repeat) + "
            "C(task_repeat) * C(prev_target_feature_match , Treatment(reference='switch_target_feature'))"
        )
        #process_subject(sub, fn = "WT", model_syntax=model2, model_tag="noRTModel")
        process_subject(sub, fn = "resRTWTT", model_syntax=model2, model_tag="noRTModel")
        #process_subject(sub, fn = "WF", model_syntax=model2, model_tag="noRTModel")
        #process_subject(sub, fn = "resRTWF", model_syntax=model2, model_tag="noRTModel")        
        
        #process_subject(sub)
