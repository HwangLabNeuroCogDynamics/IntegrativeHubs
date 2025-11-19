####################################################################
#  model regression on neural distance per subject
# - Fits OLS per searchlight sphere
# - Writes beta/t-stat maps for each contrast/effect
####################################################################

# avoid threading issues with joblib
import os
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
from patsy import build_design_matrices   #to build marginal contrast.

####################################################################
# Clean regressor names for filenames
####################################################################
def clean_regressor_name(reg_name):
    clean = reg_name
    clean = re.sub(r"C\(([^,]+),\s*Treatment\(reference=['\"]?([^'\"]+)['\"]?\)\)\[T\.([^\]]+)\]",
                   r"\1_vs_\3", clean)
    clean = re.sub(r"C\(([^)]+)\)\[T\.([^\]]+)\]", r"\1_vs_\2", clean)
    clean = re.sub(r"[^\w]+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    clean = clean.strip("_")
    return clean


####################################################################
# Paths
####################################################################
data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
out_dir  = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightNeuralDistance_RTdiff"
os.makedirs(out_dir, exist_ok=True)

mask_img_path = os.path.join(data_dir, "searchlight_mask.nii.gz")
mask_nii  = nib.load(mask_img_path)
mask_data = mask_nii.get_fdata()


####################################################################
# Spherewise regression (one sphere)
####################################################################
def regress_one_sphere(sphere_index, sdf, ds_array, model_syntax):
    """
    Fit OLS for one searchlight sphere, extract betas/t-values.
    All main_effect_* computed using **emmeans-style marginal means**. pool accors dist!
    perceptual_change_c slope uses weighting to account for balance.
    """
    sdf = sdf.copy()
    sdf["ds"] = ds_array[sphere_index, :]
    beta = {}
    tval = {}

    try:
        model = smf.ols(model_syntax, data=sdf).fit()
        params = model.params

        # raw coefficients
        beta.update(params.to_dict())
        tval.update(model.tvalues.to_dict())

        ############################################################
        # GET PARAMETER NAMES
        ############################################################
        name_pc       = "perceptual_change_c"
        name_eds_pc   = "C(hierarchical_switch_type, Treatment(reference='IDS'))[T.EDS]:perceptual_change_c"
        name_stay_pc  = "C(hierarchical_switch_type, Treatment(reference='IDS'))[T.Stay]:perceptual_change_c"

        # weighting of perceptual_change_c
        freq_tt = sdf['hierarchical_switch_type'].value_counts(normalize=True)
        p_EDS   = float(freq_tt.get('EDS', 0))
        p_Stay  = float(freq_tt.get('Stay', 0))

        ############################################################
        # update, get contrast using design-matrix averaging (not just as ref level), get design matrix out in one go.
        # this is how emmeans handel this
        ############################################################
        design_info = model.model.data.design_info

        def _mean_exog(df_mod):
            X = build_design_matrices([design_info], df_mod)[0]
            return X.mean(axis=0)

        def emm_L(level_A, level_B):
            """
            get the margianl B - A across distributions
            """
            dfA = sdf.copy()
            dfB = sdf.copy()

            for col, val in level_A.items():
                dfA[col] = val
            for col, val in level_B.items():
                dfB[col] = val

            xA = _mean_exog(dfA)
            xB = _mean_exog(dfB)
            return (xB - xA)

        ############################################################
        # Average slope for perceptual_change_c
        ############################################################
        # slope_IDS  = beta_pc
        # slope_EDS  = beta_pc + beta_eds_pc, diff from IDS ref level
        # slope_Stay = beta_pc + beta_stay_pc
        L_pc = [
            (1 if n == name_pc else
             p_EDS if n == name_eds_pc else
             p_Stay if n == name_stay_pc else
             0)
            for n in params.index
        ]
        ct = model.t_test(L_pc) #linear contrast
        beta["main_effect_perceptual_change_c"] = float(ct.effect)
        tval["main_effect_perceptual_change_c"] = float(ct.tvalue)


        ############################################################
        # slopes for perceptual_change_c
        ############################################################
        # IDS
        L_IDS = [1 if n == name_pc else 0 for n in params.index]
        ct = model.t_test(L_IDS)
        beta["perceptual_change_for_IDS"] = float(ct.effect)
        tval["perceptual_change_for_IDS"] = float(ct.tvalue)

        # EDS
        L_EDS = [1 if n in (name_pc, name_eds_pc) else 0 for n in params.index]
        ct = model.t_test(L_EDS)
        beta["perceptual_change_for_EDS"] = float(ct.effect)
        tval["perceptual_change_for_EDS"] = float(ct.tvalue)

        # Stay
        L_Stay = [1 if n in (name_pc, name_stay_pc) else 0 for n in params.index]
        ct = model.t_test(L_Stay)
        beta["perceptual_change_for_Stay"] = float(ct.effect)
        tval["perceptual_change_for_Stay"] = float(ct.tvalue)


        ############################################################
        # main effects
        ############################################################
        # EDS – IDS
        L = emm_L({'hierarchical_switch_type': 'IDS'},
                  {'hierarchical_switch_type': 'EDS'})
        ct = model.t_test(L)
        beta["main_effect_EDS_v_IDS"] = float(ct.effect)
        tval["main_effect_EDS_v_IDS"] = float(ct.tvalue)

        # IDS – Stay
        L = emm_L({'hierarchical_switch_type': 'Stay'},
                  {'hierarchical_switch_type': 'IDS'})
        ct = model.t_test(L)
        beta["main_effect_IDS_v_Stay"] = float(ct.effect)
        tval["main_effect_IDS_v_Stay"] = float(ct.tvalue)

        #response_repeat: switch – repeat
        L = emm_L({'response_repeat': 'repeat'},
                  {'response_repeat': 'switch'})
        ct = model.t_test(L)
        beta["main_effect_response_repeat"] = float(ct.effect)
        tval["main_effect_response_repeat"] = float(ct.tvalue)

        #task_repeat: switch – repeat
        L = emm_L({'task_repeat': 'repeat'},
                  {'task_repeat': 'switch'})
        ct = model.t_test(L)
        beta["main_effect_task_repeat"] = float(ct.effect)
        tval["main_effect_task_repeat"] = float(ct.tvalue)

        # prev_target_feature_switch: repeat – switch
        L = emm_L({'prev_target_feature_switch': 'switch'},
                  {'prev_target_feature_switch': 'repeat'})
        ct = model.t_test(L)
        beta["main_effect_prev_target_feature_switch"] = float(ct.effect)
        tval["main_effect_prev_target_feature_switch"] = float(ct.tvalue)

        # Errors: error – correct
        L = emm_L({'Errors': 'correct'},
                  {'Errors': 'error'})
        ct = model.t_test(L)
        beta["main_effect_Errors"] = float(ct.effect)
        tval["main_effect_Errors"] = float(ct.tvalue)

        # Prev_Errors: error – correct
        L = emm_L({'Prev_Errors': 'correct'},
                  {'Prev_Errors': 'error'})
        ct = model.t_test(L)
        beta["main_effect_Prev_Errors"] = float(ct.effect)
        tval["main_effect_Prev_Errors"] = float(ct.tvalue)

        ############################################################
        # interaction: response_repeat × task_repeat
        ############################################################
        name_rrtr = "C(response_repeat)[T.switch]:C(task_repeat)[T.switch]"
        if name_rrtr in params.index:
            L = [1 if n == name_rrtr else 0 for n in params.index]
            ct = model.t_test(L)
            beta["interaction_response_repeat_x_task_repeat"] = float(ct.effect)
            tval["interaction_response_repeat_x_task_repeat"] = float(ct.tvalue)

    except Exception:
        pass

    return beta, tval


####################################################################
# Process one subject
####################################################################
def process_subject(sub_id, fn, model_syntax, model_tag):
    print(f"→ Processing subject {sub_id}")

    subj_file = os.path.join(data_dir, f"{sub_id}_sl_adjcorr_{fn}.npy")
    if not os.path.exists(subj_file):
        print(f"✗ Missing data for {sub_id}")
        return

    ds_array = np.load(subj_file)
    n_spheres = ds_array.shape[0]

    # load behavioral data
    sdf = df[df["subject"] == int(sub_id)].copy().sort_values(["block", "trial"]).reset_index(drop=True)

    # trial filters
    sdf = sdf[(sdf['trial'] != 0) & (np.abs(sdf['zRT']) <= 3)].reset_index(drop=True)

    # categories
    for col in ["probe_repeat", "response_repeat", "task_repeat",
                "hierarchical_switch_type", "prev_target_feature_switch",
                "Errors", "Prev_Errors"]:
        sdf[col] = sdf[col].astype("category")

    # center perceptual_change
    sdf['perceptual_change_c'] = sdf['perceptual_change'] - 1.5

    # align ds_array to remaining trials
    trial_vec = (sdf['block'] - 1) * 51 + sdf['trial']
    ds_array = ds_array[:, trial_vec]

    # run regressions
    results = Parallel(n_jobs=16, verbose=5)(
        delayed(regress_one_sphere)(i, sdf, ds_array, model_syntax)
        for i in range(n_spheres)
    )

    # collect all regressor names
    reg_names = sorted(set(k for beta, _ in results for k in beta.keys()))

    # prepare storage
    betas = {r: np.full(n_spheres, np.nan) for r in reg_names}
    tvals = {r: np.full(n_spheres, np.nan) for r in reg_names}

    for i, (beta, tval) in enumerate(results):
        for r in reg_names:
            betas[r][i] = beta.get(r, np.nan)
            tvals[r][i] = tval.get(r, np.nan)

    # save
    subj_out = os.path.join(out_dir, sub_id)
    os.makedirs(subj_out, exist_ok=True)

    for r in reg_names:
        r_clean = clean_regressor_name(r)
        unmask(betas[r], mask_nii).to_filename(
            os.path.join(subj_out, f"{r_clean}_{model_tag}_{fn}_beta.nii.gz")
        )
        unmask(tvals[r], mask_nii).to_filename(
            os.path.join(subj_out, f"{r_clean}_{model_tag}_{fn}_tval.nii.gz")
        )

    print(f"✓ Finished subject {sub_id}")


####################################################################
# Load master dataframe
####################################################################
behavior_csv = "/Shared/lss_kahwang_hpc/data/HiRe/Switch_Costs/thalhi_mri_master_df.csv"
df = pd.read_csv(behavior_csv)

df['Errors'] = (df['accuracy'] != 1).astype(int).replace({1:'error',0:'correct'})
df['Prev_Errors'] = (df['prev_accuracy'] != 1).astype(int).replace({1:'error',0:'correct'})

df["response_repeat"] = df["response_switch"]
df["task_repeat"]     = df["task_switch"]
df["probe_repeat"]    = df["probe_switch"]

df["zRT"] = np.where(df["zRT"]==-99, 0, df["zRT"])
df["zRT_diff"] = 0
df["zRT_diff"][1:] = np.asarray(df["zRT"][1:]) - np.asarray(df["zRT"][:-1])


####################################################################
# Run
####################################################################
if __name__ == "__main__":
    subjects = input("Subject ID: ")

    model1 = (
        "ds ~ zRT_diff + zRT + C(Errors, Treatment(reference='correct')) + "
        "C(Prev_Errors, Treatment(reference='correct')) + "
        "C(hierarchical_switch_type, Treatment(reference='IDS')) * perceptual_change_c + "
        "C(response_repeat) * C(task_repeat) + "
        "C(task_repeat) * C(prev_target_feature_switch, Treatment(reference='switch')) + "
        "C(response_repeat) * C(prev_target_feature_switch, Treatment(reference='switch')) + "
        "C(probe_repeat)"
    )

    process_subject(subjects, fn="resRTWTT", model_syntax=model1, model_tag="ProbeFeatureRTAccuModel")
